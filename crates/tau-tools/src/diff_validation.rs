//! Fuzzy matching of LLM-produced edits against actual file content.
//!
//! Models frequently emit search/replace pairs whose `search` block is *almost*
//! a substring of the file: line numbers off by one, indentation drifted, a
//! trailing token elided, or a stale copy of a line that's since been updated.
//! Treating these as exact matches fails far too often, so this module runs a
//! cascade of progressively-more-tolerant matchers and returns the most
//! plausible replacement range.
//!
//! Matching strategies, applied in order:
//!  1. **Exact** — whole-line match.
//!  2. **Indentation-agnostic** — lines equal after `trim_start`.
//!  3. **Prefix-tail** — every line but the last matches; the last `search`
//!     line is a strict prefix of the corresponding file line. Only attempted
//!     when a line-number hint is present, since otherwise short prefixes
//!     would match too liberally.
//!  4. **Jaro-Winkler** — fuzzy similarity ≥ 0.9 over the whole window.
//!
//! Two formats are supported:
//!  - [`SearchAndReplace`] — pairs of (search, replace), with optional
//!    `{line_no}|{content}` line-number hints in the `search` block.
//!  - [`V4AHunk`] — V4A diff format with pre/post context and optional
//!    class/function `change_context` markers
//!    (<https://cookbook.openai.com/examples/gpt4-1_prompting_guide#apply-patch>).
//!
//! Originally extracted from warpdotdev/warp's `crates/ai/src/diff_validation`.

use itertools::{EitherOrBoth, Itertools};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    fmt::{self, Display},
    ops::Range,
    path::PathBuf,
    sync::LazyLock,
};
use strsim::jaro_winkler;

static LINE_NUMBER_PARSE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d+)\|(.*)$").expect("line-number regex must compile"));

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffType {
    Create {
        /// A delta for a file creation has an empty replacement line range.
        delta: DiffDelta,
    },
    Update {
        deltas: Vec<DiffDelta>,
        /// If set, the file should be renamed to this path when applying the diff.
        /// This path should also be a non-existing filepath.
        rename: Option<PathBuf>,
    },
    Delete {
        /// A delta for a file deletion has a replacement line range that spans
        /// the entire file and an empty insertion.
        delta: DiffDelta,
    },
}

impl DiffType {
    pub fn creation(content: String) -> Self {
        DiffType::Create {
            delta: DiffDelta {
                replacement_line_range: 0..0,
                insertion: content,
            },
        }
    }

    pub fn deletion(num_lines: usize) -> Self {
        DiffType::Delete {
            delta: DiffDelta {
                replacement_line_range: 1..num_lines.saturating_add(1),
                insertion: String::new(),
            },
        }
    }

    pub fn update(deltas: Vec<DiffDelta>, rename_to: Option<String>) -> Self {
        DiffType::Update {
            deltas,
            rename: rename_to.map(Into::into),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AIRequestedCodeDiff {
    pub file_name: String,
    pub diff_type: DiffType,
    /// Failure types worth surfacing to the caller (logging, retry, telemetry).
    pub failures: Option<DiffMatchFailures>,
    /// Original file content read during diff matching.
    /// Populated for edits and deletes; empty for new file creation.
    pub original_content: String,
}

// `failures` and `original_content` are intentionally excluded from equality:
// they describe *how* a diff was produced, not what it does.
impl PartialEq for AIRequestedCodeDiff {
    fn eq(&self, other: &Self) -> bool {
        self.file_name == other.file_name && self.diff_type == other.diff_type
    }
}
impl Eq for AIRequestedCodeDiff {}

impl AIRequestedCodeDiff {
    /// Determines if the failures are severe enough to warrant some logging/remediation.
    pub fn warrants_failure(&self) -> bool {
        match &self.failures {
            // NOTE: Avoid `..` rest patterns here so that future fields force a
            // deliberate choice about how they affect retries.
            Some(DiffMatchFailures {
                fuzzy_match_failures,
                noop_deltas,
                missing_line_numbers: _,
            }) => {
                let update_deltas_empty = match &self.diff_type {
                    DiffType::Update { deltas, .. } => deltas.is_empty(),
                    DiffType::Create { .. } | DiffType::Delete { .. } => false,
                };

                *fuzzy_match_failures > 0 || (*noop_deltas > 0 && update_deltas_empty)
            }
            None => false,
        }
    }
}

/// One resolved hunk: replace `replacement_line_range` (1-indexed, end-exclusive)
/// with `insertion`. An empty range `0..0` means prepend at the start of the file.
#[derive(Clone, PartialEq, Eq)]
pub struct DiffDelta {
    pub replacement_line_range: Range<usize>,
    pub insertion: String,
}

impl fmt::Debug for DiffDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if cfg!(debug_assertions) {
            write!(
                f,
                "DiffDelta {{\nreplacement_line_range: {:?},",
                &self.replacement_line_range
            )?;
            f.write_str("\n--insertion--\n")?;
            f.write_str(&self.insertion)?;
            f.write_str("\n}")
        } else {
            Ok(())
        }
    }
}

/// A search/replace pair as produced by an LLM. The `search` block may carry
/// `{line_no}|{content}` line-number hints which improve disambiguation when
/// the same text appears multiple times.
#[cfg_attr(test, derive(PartialEq))]
pub struct SearchAndReplace {
    pub search: String,
    pub replace: String,
}

#[cfg(debug_assertions)]
impl fmt::Debug for SearchAndReplace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SearchAndReplace {\n--search--\n")?;
        f.write_str(&self.search)?;
        f.write_str("\n--replace--\n")?;
        f.write_str(&self.replace)?;
        f.write_str("\n}")
    }
}

/// V4A diff hunk. See
/// <https://cookbook.openai.com/examples/gpt4-1_prompting_guide#apply-patch>.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct V4AHunk {
    /// Class/function markers (`@@` lines) that narrow the search region.
    pub change_context: Vec<String>,
    /// Context lines immediately before the change.
    pub pre_context: String,
    /// Old content being replaced. Empty if this hunk only adds lines.
    pub old: String,
    /// New content. Empty if this hunk only deletes lines.
    pub new: String,
    /// Context lines immediately after the change.
    pub post_context: String,
}

/// A customized version of [`str::lines`] which treats the empty string differently.
///
/// Generally, calling `.lines()` (on UNIX) is equivalent to calling `.split("\n")`. However,
/// trailing empty lines are ignored by [`str::lines`] which produces some weird behavior.
fn lines(s: &str) -> impl Iterator<Item = &str> {
    match s {
        "" => "\n".lines(),
        _ => s.lines(),
    }
}

/// Returns `true` if `search` is a non-empty strict prefix of `file_line` after trimming
/// leading whitespace from both.
///
/// "Strict" means the trimmed `file_line` is strictly longer than the trimmed `search`;
/// equal-length matches are rejected so callers can rely on there being a non-empty unmatched
/// suffix.
fn is_strict_trimmed_prefix(search: &str, file_line: &str) -> bool {
    let trimmed_search = search.trim_start();
    let trimmed_file = file_line.trim_start();
    !trimmed_search.is_empty()
        && trimmed_file.len() > trimmed_search.len()
        && trimmed_file.starts_with(trimmed_search)
}

/// If `search_line` is a proper prefix of `file_line` (ignoring leading whitespace on both),
/// returns the unmatched suffix from `file_line`. Otherwise returns `None`.
fn unmatched_line_suffix<'a>(search_line: &str, file_line: &'a str) -> Option<&'a str> {
    if is_strict_trimmed_prefix(search_line, file_line) {
        let trimmed_search = search_line.trim_start();
        let trimmed_file = file_line.trim_start();
        Some(&trimmed_file[trimmed_search.len()..])
    } else {
        None
    }
}

/// Strip leading `{number}|` markers from each line. Models are instructed not
/// to include line numbers in the *replacement* block, but they sometimes do.
pub fn remove_extra_line_num_prefix(replace: String) -> String {
    static LINE_NUMBER_PATTERN: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^\d+\|").expect("line number regex must compile"));

    lines(&replace)
        .map(|line| LINE_NUMBER_PATTERN.replace(line, "").into_owned())
        .join("\n")
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DiffMatchFailures {
    /// Failures to perform a fuzzy match with content.
    pub fuzzy_match_failures: u8,
    /// The <search> and <replace> content was identical.
    pub noop_deltas: u8,
    /// Search blocks that are missing line numbers.
    pub missing_line_numbers: u8,
}

/// Resolve a list of search/replace edits against `file_content` using the
/// fuzzy matching cascade described at the module level.
///
/// Method, for each search block of n lines:
/// - Scan through n-sized windows of the file.
/// - Score each window with the active strategy.
/// - If a line-number hint was present, prefer the match closest to it;
///   otherwise prefer the highest-similarity match.
pub fn fuzzy_match_diffs(
    file_name: &str,
    diffs: &[SearchAndReplace],
    file_content: impl Into<String>,
) -> AIRequestedCodeDiff {
    let file_content = file_content.into();
    let (deltas, failures) = fuzzy_match_file_diffs(diffs, &file_content);

    // Only surface failures when they are meaningful to the caller.
    // In particular, noop diffs are only considered failures if they result in no applied deltas.
    let update_deltas_empty = deltas.is_empty();
    let failures = if failures.fuzzy_match_failures > 0
        || failures.missing_line_numbers > 0
        || (failures.noop_deltas > 0 && update_deltas_empty)
    {
        Some(failures)
    } else {
        None
    };

    AIRequestedCodeDiff {
        file_name: file_name.into(),
        diff_type: DiffType::Update {
            deltas,
            rename: None,
        },
        failures,
        original_content: file_content,
    }
}

/// Resolve V4A-format hunks against `file_content`. Tries exact, then
/// indentation-agnostic, then Jaro-Winkler matching.
pub fn fuzzy_match_v4a_diffs(
    file_name: &str,
    diffs: &[V4AHunk],
    rename_to: Option<String>,
    file_content: impl Into<String>,
) -> AIRequestedCodeDiff {
    let file_content = file_content.into();
    let mut deltas = Vec::new();
    let mut failures = DiffMatchFailures::default();

    let file_lines: Vec<&str> = file_content.lines().collect();

    for diff in diffs {
        if diff.old == diff.new {
            tracing::info!("Ignoring V4A diff with identical old and new content.");
            failures.noop_deltas += 1;
            continue;
        }

        let match_range = find_v4a_match(diff, &file_lines);

        match match_range {
            Some(range) => {
                let matched_content = file_lines[range.start - 1..range.end - 1].join("\n");
                if diff.new == matched_content {
                    tracing::info!(
                        "Ignoring V4A diff where new content is identical to matched file content"
                    );
                    failures.noop_deltas += 1;
                    continue;
                }

                deltas.push(DiffDelta {
                    replacement_line_range: range.start..range.end,
                    insertion: diff.new.clone(),
                });
            }
            None => {
                tracing::warn!("Failed to find matching location for V4A diff");
                failures.fuzzy_match_failures += 1;
            }
        }
    }

    let update_deltas_empty = deltas.is_empty();
    let failures = if failures.fuzzy_match_failures > 0
        || failures.missing_line_numbers > 0
        || (failures.noop_deltas > 0 && update_deltas_empty)
    {
        Some(failures)
    } else {
        None
    };

    AIRequestedCodeDiff {
        file_name: file_name.into(),
        diff_type: DiffType::update(deltas, rename_to),
        failures,
        original_content: file_content,
    }
}

fn fuzzy_match_file_diffs(
    diffs: &[SearchAndReplace],
    file_content: &str,
) -> (Vec<DiffDelta>, DiffMatchFailures) {
    let mut deltas = Vec::new();
    let mut failures = DiffMatchFailures::default();

    let target_lines: Vec<&str> = lines(file_content).collect();

    for diff in diffs {
        #[cfg(debug_assertions)]
        tracing::debug!("{diff:#?}");

        let (mut line_range, search) = parse_line_numbers(&diff.search);

        // Missing line numbers are not necessarily fatal, due to fuzzy matching, but we still
        // want to track them.
        if line_range.is_none() && !search.is_empty() {
            failures.missing_line_numbers += 1;
        }

        if search == diff.replace {
            tracing::info!("Ignoring diff with identical <search> and <replace>.");
            failures.noop_deltas += 1;
            continue;
        }

        // Find similar sections in the file content using the matching strategies.
        let fuzzy_match_line_numbers = if line_range == Some(0..0) {
            // An empty line range indicates prepending to the file.
            line_range
        } else {
            line_range = line_range.filter(|range| {
                tracing::debug!("Parsed line range: {range:?}");
                range.start > 0
                    && range.start <= target_lines.len()
                    && range.end > 0
                    // Because the end is both 1-indexed and exclusive, the last valid end is 1 past
                    // the last line number.
                    && range.end <= target_lines.len() + 1
                    && range.end >= range.start
            });

            // First, search for an exact match, then fall back to ignoring whitespace if needed.
            let mut matched = match_diff(
                &search,
                line_range.clone(),
                &target_lines,
                SECTION_MATCH_THRESHOLD,
                MakeExactMatch,
            )
            .or_else(|| {
                match_diff(
                    &search,
                    line_range.clone(),
                    &target_lines,
                    SECTION_MATCH_THRESHOLD,
                    MakeIndentationAgnosticMatch,
                )
            });

            // Prefix-tail rescue: only attempt when we have a line-number hint to
            // disambiguate. Without a hint, short prefix searches like `fn main() {`
            // could match many windows in the file and silently pick the wrong one.
            if matched.is_none() && line_range.is_some() {
                matched = match_diff(
                    &search,
                    line_range.clone(),
                    &target_lines,
                    // Binary scorer: match is exact (1.0) or not at all.
                    1.0,
                    MakePrefixTailMatch,
                );
            }

            if matched.is_none() {
                matched = match_diff(
                    &search,
                    line_range.clone(),
                    &target_lines,
                    SECTION_MATCH_THRESHOLD,
                    MakeJaroWinklerMatch,
                );
            }

            matched
        };

        tracing::debug!("fuzzy match result: {fuzzy_match_line_numbers:?}");

        match fuzzy_match_line_numbers {
            Some(range) => {
                #[cfg(debug_assertions)]
                {
                    tracing::debug!("Matched content in file:");
                    for line_num in range.clone() {
                        tracing::debug!("{}|{}", line_num, target_lines[line_num - 1]);
                    }
                }

                // Check if the text to be replaced is identical to the replacement block.
                // This may happen if the search block was based on stale file information, and the
                // replacement block matches the file content but _not_ the search block.
                if range != (0..0)
                    && diff
                        .replace
                        .lines()
                        .zip_longest(&target_lines[range.start - 1..range.end - 1])
                        .all(|pair| match pair {
                            EitherOrBoth::Both(replace, original) => replace == *original,
                            EitherOrBoth::Left(_) | EitherOrBoth::Right(_) => false,
                        })
                {
                    tracing::info!("Ignoring diff with <replace> identical to the file contents");
                    failures.noop_deltas += 1;
                    continue;
                }

                // Some LLMs emit a search block whose last line is a prefix of the
                // actual file line (e.g. search ends with "let x" while the file has
                // "let x = 2;"). Because the matcher operates on whole-line windows,
                // the delta would replace the entire line and drop the unmatched
                // suffix. Detect this and preserve the suffix in the insertion.
                let mut insertion = diff.replace.clone();
                if range.end >= 2 && lines(&search).count() == lines(&insertion).count() {
                    if let Some(suffix) = lines(&search)
                        .last()
                        .and_then(|last| unmatched_line_suffix(last, target_lines[range.end - 2]))
                    {
                        insertion.push_str(suffix);
                    }
                }
                deltas.push(DiffDelta {
                    replacement_line_range: range.start..range.end,
                    insertion,
                });
            }
            None => {
                failures.fuzzy_match_failures += 1;
            }
        }
    }

    (deltas, failures)
}

#[derive(Debug, PartialEq)]
pub struct Match {
    pub start_line: usize,
    pub end_line: usize,
    pub similarity: f64,
}

/// Find similar sections in the target file using Jaro-Winkler similarity.
/// Returned line numbers are 1-indexed.
pub fn find_similar_sections(
    search_text: &str,
    target_lines: &[&str],
    threshold: f64,
) -> Vec<Match> {
    let search_len = search_text.lines().count();
    if search_len == 0 {
        return Vec::new();
    }
    target_lines
        .windows(search_len)
        .enumerate()
        .filter_map(|(i, target_window)| {
            let similarity = section_similarity(search_text, target_window);
            (similarity >= threshold).then_some(Match {
                start_line: i + 1,
                similarity,
                end_line: i + search_len + 1,
            })
        })
        .collect()
}

/// Scores `search_window_lines`-length windows using a provided scoring function.
///
/// Returned matches are 1-indexed, and sorted by similarity.
/// If `expected_range` is provided, the scoring also considers how close matches are to the expected range.
fn score_matches<T: Scorer>(
    target_lines: &[&str],
    search_window_lines: usize,
    threshold: f64,
    expected_range: Option<Range<usize>>,
    scorer: &T,
) -> Vec<Match> {
    if search_window_lines == 0 || search_window_lines > target_lines.len() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    let mut max_similarity = 0.;
    #[cfg(debug_assertions)]
    let mut most_similar_range = None;
    for (i, window) in target_lines.windows(search_window_lines).enumerate() {
        let similarity = scorer.score(window);
        if similarity > max_similarity {
            max_similarity = similarity;
            #[cfg(debug_assertions)]
            {
                most_similar_range = Some(i..i + search_window_lines);
            }
        }
        if similarity >= threshold {
            matches.push(Match {
                start_line: i + 1,
                end_line: i + search_window_lines + 1,
                similarity,
            });
        }
    }

    if matches.is_empty() {
        tracing::debug!("No matches meeting the threshold for scorer {scorer}");
        #[cfg(debug_assertions)]
        if let Some(range) = most_similar_range {
            tracing::debug!(
                "Closest match with score {}:\n{}",
                max_similarity,
                &target_lines[range].join("\n")
            );
        }
    }

    matches.sort_by(move |a, b| {
        let by_similarity = a
            .similarity
            .partial_cmp(&b.similarity)
            .unwrap_or(Ordering::Equal)
            .reverse();
        if let Some(Range { start, .. }) = expected_range {
            by_similarity.then_with(|| {
                let a_distance = a.start_line.abs_diff(start);
                let b_distance = b.start_line.abs_diff(start);
                a_distance.cmp(&b_distance)
            })
        } else {
            by_similarity
        }
    });

    matches
}

/// A `Scorer` scores a target window by how closely it matches some search text. Higher scores indicate closer matches.
trait Scorer: fmt::Display {
    fn score(&self, target_lines: &[&str]) -> f64;
}

/// Factory trait for [`Scorer`]s. Works around lifetime issues with scorers
/// that reference the search text.
trait MakeScorer: fmt::Display {
    type ScorerInstance<'a>: Scorer;
    fn for_search<'a>(&self, search_text: &'a str) -> Self::ScorerInstance<'a>;
}

/// Returns 1 iff the target window equals the search block on every line.
struct ExactMatch<'a> {
    search_lines: Vec<&'a str>,
}

impl<'a> ExactMatch<'a> {
    fn new(search_text: &'a str) -> Self {
        let search_lines = lines(search_text).collect();
        Self { search_lines }
    }
}

impl Scorer for ExactMatch<'_> {
    fn score(&self, target_lines: &[&str]) -> f64 {
        if target_lines == self.search_lines {
            1.0
        } else {
            0.0
        }
    }
}

impl fmt::Display for ExactMatch<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&MakeExactMatch, f)
    }
}

#[derive(Clone)]
struct MakeExactMatch;

impl Display for MakeExactMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Exact")
    }
}

impl MakeScorer for MakeExactMatch {
    type ScorerInstance<'a> = ExactMatch<'a>;

    fn for_search<'a>(&self, search_text: &'a str) -> Self::ScorerInstance<'a> {
        ExactMatch::new(search_text)
    }
}

/// Returns 1 iff the target window equals the search block on every line after `trim_start`.
struct IndentationAgnosticMatch<'a> {
    search_lines: Vec<&'a str>,
}

impl<'a> IndentationAgnosticMatch<'a> {
    fn new(search_text: &'a str) -> Self {
        let search_lines = lines(search_text).map(|line| line.trim_start()).collect();
        Self { search_lines }
    }
}

impl Scorer for IndentationAgnosticMatch<'_> {
    fn score(&self, target_lines: &[&str]) -> f64 {
        debug_assert_eq!(
            self.search_lines.len(),
            target_lines.len(),
            "Incorrect target window length"
        );

        if target_lines
            .iter()
            .map(|line| line.trim_start())
            .zip(self.search_lines.iter())
            .all(|(a, b)| a == *b)
        {
            1.0
        } else {
            0.0
        }
    }
}

impl fmt::Display for IndentationAgnosticMatch<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&MakeIndentationAgnosticMatch, f)
    }
}

#[derive(Clone)]
struct MakeIndentationAgnosticMatch;

impl Display for MakeIndentationAgnosticMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Indentation Agnostic")
    }
}

impl MakeScorer for MakeIndentationAgnosticMatch {
    type ScorerInstance<'a> = IndentationAgnosticMatch<'a>;

    fn for_search<'a>(&self, search_text: &'a str) -> Self::ScorerInstance<'a> {
        IndentationAgnosticMatch::new(search_text)
    }
}

/// Rescues diffs whose final search line is a strict prefix of the actual
/// final file line (e.g. `if foo() {` when the file has `if foo() && bar() {`).
/// Jaro-Winkler often scores such pairs just under the 0.9 threshold for long
/// lines; this scorer handles the case deterministically when a line-number
/// hint is available to disambiguate.
struct PrefixTailMatch<'a> {
    /// Lines with leading whitespace trimmed.
    search_lines: Vec<&'a str>,
}

impl<'a> PrefixTailMatch<'a> {
    fn new(search_text: &'a str) -> Self {
        let search_lines = lines(search_text).map(|line| line.trim_start()).collect();
        Self { search_lines }
    }
}

impl Scorer for PrefixTailMatch<'_> {
    fn score(&self, target_lines: &[&str]) -> f64 {
        if target_lines.len() != self.search_lines.len() || self.search_lines.is_empty() {
            return 0.0;
        }

        let last_idx = self.search_lines.len() - 1;

        let prefix_lines_exact = self.search_lines[..last_idx]
            .iter()
            .zip(&target_lines[..last_idx])
            .all(|(s, t)| *s == t.trim_start());
        if !prefix_lines_exact {
            return 0.0;
        }

        if is_strict_trimmed_prefix(self.search_lines[last_idx], target_lines[last_idx]) {
            1.0
        } else {
            0.0
        }
    }
}

impl fmt::Display for PrefixTailMatch<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&MakePrefixTailMatch, f)
    }
}

#[derive(Clone)]
struct MakePrefixTailMatch;

impl Display for MakePrefixTailMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Prefix Tail")
    }
}

impl MakeScorer for MakePrefixTailMatch {
    type ScorerInstance<'a> = PrefixTailMatch<'a>;

    fn for_search<'a>(&self, search_text: &'a str) -> Self::ScorerInstance<'a> {
        PrefixTailMatch::new(search_text)
    }
}

fn section_similarity(search_text: &str, target_lines: &[&str]) -> f64 {
    let window_text = target_lines.join("\n");
    jaro_winkler(search_text, &window_text)
}

const SECTION_MATCH_THRESHOLD: f64 = 0.9;

/// Given a search block and a scoring function, find the most likely matching
/// range of lines from the file.
///
/// The result is 1-indexed. An empty search corresponds to `0..0` — prepending
/// to the file.
fn match_diff<S: MakeScorer>(
    search: &str,
    line_range: Option<Range<usize>>,
    file_content: &[&str],
    threshold: f64,
    factory: S,
) -> Option<Range<usize>> {
    let search_length = lines(search).count();
    let scorer = factory.for_search(search);

    // If we could parse a line range, check if it's approximately correct.
    if let Some(Range { start, end }) = &line_range {
        let search_start = start.saturating_sub(2);
        let search_end = (end + 2).min(file_content.len());
        if search_start <= search_end {
            let local_lines = &file_content[search_start..search_end];
            let local_matches = score_matches(
                local_lines,
                search_length,
                threshold,
                line_range
                    .clone()
                    .map(|range| range.start - search_start..range.end - search_start),
                &scorer,
            );
            if let Some(local_match) = local_matches.first() {
                let local_start = local_match.start_line + search_start;
                let local_end = local_match.end_line + search_start;
                tracing::debug!(
                    "Line numbers approximately correct. Parsed: {start}-{end} Matched {local_start}-{local_end} with {factory}",
                );
                return Some(local_start..local_end);
            }
        }
    }

    let matches = score_matches(
        file_content,
        search_length,
        threshold,
        line_range.clone(),
        &scorer,
    );
    if let Some(m) = matches.first() {
        match line_range {
            Some(Range { start, end }) => {
                tracing::debug!(
                    "Mismatched line numbers fixed by matching. Parsed: {start}-{end} Matched {}-{} with {factory}",
                    m.start_line,
                    m.end_line
                );
            }
            None => {
                tracing::debug!("Missing line numbers fixed by matching with {factory}");
            }
        }
        return Some(m.start_line..m.end_line);
    }
    None
}

/// Parse the optional `{line_no}|{content}` prefix from each line of `search`.
///
/// Returns the parsed line range (1-indexed, end-exclusive) and the search
/// content with prefixes stripped. An empty `search` returns `Some(0..0)`,
/// which signals "prepend to the file".
pub fn parse_line_numbers(search: &str) -> (Option<Range<usize>>, String) {
    let parsed: Vec<_> = search.lines().map(parse_line_number).collect();
    if parsed.is_empty() {
        (Some(0..0), search.to_string())
    } else {
        let starting_index = parsed.first().expect("We checked there is a line").0;
        let ending_index = parsed.last().expect("We checked there is a line").0;
        match (starting_index, ending_index) {
            (Some(start), Some(end)) => (
                Some(start..end + 1),
                parsed
                    .iter()
                    .map(|(_, line)| *line)
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => (None, search.to_string()),
        }
    }
}

fn parse_line_number(search: &str) -> (Option<usize>, &str) {
    if let Some((line_number, line)) = LINE_NUMBER_PARSE
        .captures(search)
        .and_then(|m| try_tuple2(m.get(1), m.get(2)))
        .and_then(|(a, b)| Some((a.as_str().parse::<usize>().ok()?, b.as_str())))
    {
        (Some(line_number), line)
    } else {
        (None, search)
    }
}

/// Jaro-Winkler over the indentation-stripped joined-line text.
struct JaroWinklerScorer {
    search_text: String,
}

impl JaroWinklerScorer {
    fn new(search_text: &str) -> Self {
        let search_text = lines(search_text)
            .map(str::trim_start)
            .collect_vec()
            .join("\n");
        Self { search_text }
    }
}

impl Scorer for JaroWinklerScorer {
    fn score(&self, target_lines: &[&str]) -> f64 {
        let target_text = target_lines
            .iter()
            .map(|line| line.trim_start())
            .collect_vec()
            .join("\n");
        jaro_winkler(&self.search_text, &target_text)
    }
}

impl fmt::Display for JaroWinklerScorer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&MakeJaroWinklerMatch, f)
    }
}

#[derive(Clone)]
struct MakeJaroWinklerMatch;

impl Display for MakeJaroWinklerMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Jaro-Winkler")
    }
}

impl MakeScorer for MakeJaroWinklerMatch {
    type ScorerInstance<'a> = JaroWinklerScorer;

    fn for_search<'a>(&self, _search_text: &'a str) -> Self::ScorerInstance<'a> {
        JaroWinklerScorer::new(_search_text)
    }
}

fn try_tuple2<A, B>(a: Option<A>, b: Option<B>) -> Option<(A, B)> {
    match (a, b) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

/// Locate a V4A hunk in the file. Returns a 1-indexed line range covering the
/// `old` content (or the insertion point if `old` is empty).
fn find_v4a_match(edit: &V4AHunk, file_lines: &[&str]) -> Option<Range<usize>> {
    let pre_context_lines: Vec<&str> = edit.pre_context.lines().collect();
    let old_lines: Vec<&str> = edit.old.lines().collect();
    let post_context_lines: Vec<&str> = edit.post_context.lines().collect();

    let search_start = if !edit.change_context.is_empty() {
        find_change_context_start(&edit.change_context, file_lines)?
    } else {
        0
    };

    let search_lines = &file_lines[search_start..];

    let pattern_length = pre_context_lines.len() + old_lines.len() + post_context_lines.len();
    if pattern_length == 0 {
        return Some((search_start + 1)..(search_start + 1));
    }
    if pattern_length > search_lines.len() {
        return None;
    }

    let combined_search = [
        pre_context_lines.as_slice(),
        old_lines.as_slice(),
        post_context_lines.as_slice(),
    ]
    .concat()
    .join("\n");

    if let Some(range) = match_diff(
        &combined_search,
        None,
        search_lines,
        1.0,
        MakeExactMatch,
    ) {
        return calculate_old_range(search_start, range, &pre_context_lines, &old_lines);
    }

    if let Some(range) = match_diff(
        &combined_search,
        None,
        search_lines,
        1.0,
        MakeIndentationAgnosticMatch,
    ) {
        tracing::debug!("V4A match found using indentation-agnostic matching");
        return calculate_old_range(search_start, range, &pre_context_lines, &old_lines);
    }

    if let Some(range) = match_diff(
        &combined_search,
        None,
        search_lines,
        SECTION_MATCH_THRESHOLD,
        MakeJaroWinklerMatch,
    ) {
        tracing::debug!("V4A match found using JaroWinkler fuzzy matching");
        return calculate_old_range(search_start, range, &pre_context_lines, &old_lines);
    }

    None
}

/// Calculate the line range for the old content (or insertion point if old is empty).
/// Returns 1-indexed line range.
fn calculate_old_range(
    search_start: usize,
    matched_range: Range<usize>,
    pre_context_lines: &[&str],
    old_lines: &[&str],
) -> Option<Range<usize>> {
    let old_start = search_start + matched_range.start - 1 + pre_context_lines.len();
    let old_end = old_start + old_lines.len();

    Some((old_start + 1)..(old_end + 1))
}

/// Find the starting line for searching based on change-context markers
/// (class/function signatures). Returns a 0-indexed line number.
fn find_change_context_start(change_context: &[String], file_lines: &[&str]) -> Option<usize> {
    let mut current_pos = 0;

    for marker in change_context {
        if marker.is_empty() {
            continue;
        }

        let relative_match = file_lines[current_pos..]
            .iter()
            .position(|line| line.trim_start().starts_with(marker.trim()))?;
        current_pos = current_pos + relative_match + 1;
    }

    Some(current_pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    fn deltas(diff: &AIRequestedCodeDiff) -> &[DiffDelta] {
        match &diff.diff_type {
            DiffType::Update { deltas, .. } => deltas,
            other => panic!("Expected Update diff_type, got {other:?}"),
        }
    }

    const CONTENT: &str = "I'd just like to interject
                        for a moment. What you're refering to as
                        Linux, is in fact, GNU/Linux, or as I've
                        recently taken to calling it, GNU plus
                        Linux. Linux is not an operating system
                        unto itself, but rather another free
                        component of a fully functioning GNU
                        system made useful by the GNU corelibs,
                        shell utilities and vital system
                        components comprising a full OS as
                        defined by POSIX.";

    #[test]
    fn test_simple() {
        let input_diffs = vec![
            SearchAndReplace {
                search: "2|hey".to_string(),
                replace: "what".to_string(),
            },
            SearchAndReplace {
                search: "4|world\n5|of".to_string(),
                replace: "hey".to_string(),
            },
        ];

        let diff = fuzzy_match_diffs("test.rs", &input_diffs, "what\nhey\nthere\nworld\nof\n");
        assert_eq!(diff.file_name, "test.rs");
        assert_eq!(
            deltas(&diff),
            &[
                DiffDelta {
                    replacement_line_range: 2..3,
                    insertion: "what".to_string(),
                },
                DiffDelta {
                    replacement_line_range: 4..6,
                    insertion: "hey".to_string(),
                }
            ]
        );
    }

    #[test]
    fn test_incorrect_line_numbers() {
        let input_diffs = vec![SearchAndReplace {
            search: "4|world\n5|of".to_string(),
            replace: "hey".to_string(),
        }];

        let diff = fuzzy_match_diffs("test.rs", &input_diffs, "what\nthere\nworld\nof");
        assert_eq!(diff.file_name, "test.rs");
        assert_eq!(
            deltas(&diff),
            &[DiffDelta {
                replacement_line_range: 3..5,
                insertion: "hey".to_string(),
            }]
        );
    }

    #[test]
    fn test_missing_line_numbers() {
        let input_diffs = vec![SearchAndReplace {
            search: "hey\nthere".to_string(),
            replace: "world".to_string(),
        }];

        let diff = fuzzy_match_diffs("test.rs", &input_diffs, "what\nhey\nthere\nworld\nof\n");
        assert_eq!(diff.file_name, "test.rs");
        assert_eq!(
            deltas(&diff),
            &[DiffDelta {
                replacement_line_range: 2..4,
                insertion: "world".to_string(),
            }]
        );

        let failures = diff.failures.expect("Expected failures to be tracked");
        assert_eq!(failures.missing_line_numbers, 1);
        assert_eq!(failures.fuzzy_match_failures, 0);
        assert_eq!(failures.noop_deltas, 0);
    }

    #[test]
    fn test_blank_search() {
        let input_diffs = vec![SearchAndReplace {
            search: "".to_string(),
            replace: "hey".to_string(),
        }];

        let diff = fuzzy_match_diffs("test.rs", &input_diffs, "what\nhey\nthere\nworld\nof\n");
        assert_eq!(diff.file_name, "test.rs");
        assert_eq!(
            deltas(&diff),
            &[DiffDelta {
                replacement_line_range: 0..0,
                insertion: "hey".to_string(),
            }]
        );
    }

    #[test]
    fn test_closest() {
        let input_diffs = vec![SearchAndReplace {
            search: "4|world\n5|of".to_string(),
            replace: "hey".to_string(),
        }];

        let diff = fuzzy_match_diffs(
            "test.rs",
            &input_diffs,
            "what\nhey\nworld\nof\nthe\nworld\nof\n",
        );
        assert_eq!(diff.file_name, "test.rs");
        assert_eq!(
            deltas(&diff),
            &[DiffDelta {
                replacement_line_range: 3..5,
                insertion: "hey".to_string(),
            }]
        );
    }

    #[test]
    fn test_line_numbers_off_by_one() {
        let insertion = "                        Linux, is in fact, GNU/Linux, or as I've
                        recently taken to calling it, GNU plus
                        Linux. Linux is not an operating system
                        unto itself, but rather another free
                        component of a fully functioning GNU
                        system made useful by the GNU corelibs,
                        hello, world!"
            .to_string();
        let input_diffs = vec![SearchAndReplace {
            search: "2|                        Linux, is in fact, GNU/Linux, or as I've\n\
                     3|                        recently taken to calling it, GNU plus\n\
                     4|                        Linux. Linux is not an operating system\n\
                     5|                        unto itself, but rather another free\n\
                     6|                        component of a fully functioning GNU\n\
                     7|                        system made useful by the GNU corelibs,"
                .to_string(),
            replace: insertion.clone(),
        }];
        let diff = fuzzy_match_diffs("test.rs", &input_diffs, CONTENT);
        assert_eq!(
            deltas(&diff),
            &[DiffDelta {
                replacement_line_range: 3..9,
                insertion,
            }]
        );
    }

    #[test]
    fn test_append_to_end_of_file() {
        let input_diffs = vec![SearchAndReplace {
            search: "3|".to_string(),
            replace: "foo".to_string(),
        }];
        let diff = fuzzy_match_diffs("test.rs", &input_diffs, "\n\n\n");
        assert_eq!(
            deltas(&diff),
            &[DiffDelta {
                replacement_line_range: 3..4,
                insertion: "foo".to_string(),
            }]
        )
    }

    #[test]
    fn test_totally_unrelated_search() {
        let input_diffs = vec![SearchAndReplace {
            search: "4|foo bar baz".to_string(),
            replace: "hello, world!".to_string(),
        }];
        let diff = fuzzy_match_diffs("test.rs", &input_diffs, CONTENT);
        assert!(deltas(&diff).is_empty());
        assert!(diff.failures.is_some());
    }

    /// The agent sometimes emits a search whose final line is a prefix of the actual file line.
    /// Before `PrefixTailMatch`, the Jaro-Winkler scorer landed just under the 0.9 threshold for
    /// long lines and the diff failed. With `PrefixTailMatch` in the cascade, the rescue
    /// succeeds and the existing suffix-preservation fixup splices the unmatched tail into
    /// the insertion.
    #[test]
    fn test_prefix_tail_rescue_with_line_number_hint() {
        let actual_line = "if the stripping tool encounters any error (nesting, unmatched markers, UTF-8 decode failure), the sync workflow **fails** and does **not** update the watermark.  the next run will retry from the same commit.  this is correct fail-closed behavior \u{2014} a stripping error might indicate a condition that could cause private code to leak.";
        let file_content = format!("(preamble)\n\n### error handling\n\n{actual_line}\n\n(trailer)\n");

        let search = "5|if the stripping tool encounters any error (nesting, unmatched markers, UTF-8 decode failure), the sync workflow **fails** and does **not** update the watermark.";
        let replace = "if the stripping tool encounters any error (nesting, unmatched markers, UTF-8 decode failure, symlinks), the sync workflow **fails** and does **not** update the watermark.";

        let input_diffs = vec![SearchAndReplace {
            search: search.to_string(),
            replace: replace.to_string(),
        }];

        let diff = fuzzy_match_diffs("TECH-DESIGN.md", &input_diffs, &file_content);

        let unmatched_suffix = &actual_line[search.strip_prefix("5|").unwrap().len()..];
        let expected_insertion = format!("{replace}{unmatched_suffix}");
        assert_eq!(
            deltas(&diff),
            &[DiffDelta {
                replacement_line_range: 5..6,
                insertion: expected_insertion,
            }]
        );

        assert!(diff.failures.is_none());
        assert!(!diff.warrants_failure());
    }

    #[test]
    fn test_parse_line_numbers() {
        let search = "1|hey\n2|there\n3|world";
        let (line_range, line) = parse_line_numbers(search);
        assert_eq!(line_range, Some(1..4));
        assert_eq!(line, "hey\nthere\nworld");

        let search = "hey\nthere";
        let (line_range, line) = parse_line_numbers(search);
        assert_eq!(line_range, None);
        assert_eq!(line, "hey\nthere");

        let search = "";
        let (line_range, line) = parse_line_numbers(search);
        assert_eq!(line_range, Some(0..0));
        assert_eq!(line, "");
    }

    #[test]
    fn test_remove_extra_line_num_prefix() {
        let input = "1|first line\n2|second line\n3|third line".to_string();
        assert_eq!(
            remove_extra_line_num_prefix(input),
            "first line\nsecond line\nthird line"
        );

        let input = "first line\nsecond line".to_string();
        assert_eq!(
            remove_extra_line_num_prefix(input),
            "first line\nsecond line"
        );

        assert_eq!(remove_extra_line_num_prefix("".to_string()), "");

        assert_eq!(
            remove_extra_line_num_prefix("1|only line".to_string()),
            "only line"
        );

        let input = "first line\n2|second line\n3|third line".to_string();
        assert_eq!(
            remove_extra_line_num_prefix(input),
            "first line\nsecond line\nthird line"
        );

        let input = "no number line".to_string();
        assert_eq!(remove_extra_line_num_prefix(input.clone()), input);
    }

    #[test]
    fn test_find_similar_sections_out_of_bounds() {
        let matches = find_similar_sections("hey\nthere\nyou", &[], 0.9);
        assert!(matches.is_empty());

        let matches = find_similar_sections("hey\nthere\nyou", &["hey", "there", "you"], 0.9);
        assert_eq!(
            matches,
            vec![Match {
                start_line: 1,
                end_line: 4,
                similarity: 1.0
            }]
        );

        let matches = find_similar_sections("hey\nthere\nyou", &["hey", "there"], 0.9);
        assert!(matches.is_empty());

        let matches = find_similar_sections("", &[], 0.9);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_v4a_exact_match() {
        let hunks = vec![V4AHunk {
            change_context: vec![],
            pre_context: "fn main() {".to_string(),
            old: "    println!(\"Hello\");".to_string(),
            new: "    println!(\"Hello, World!\");".to_string(),
            post_context: "}".to_string(),
        }];

        let file_content = "fn main() {\n    println!(\"Hello\");\n}";
        let diff = fuzzy_match_v4a_diffs("test.rs", &hunks, None, file_content);

        assert_eq!(diff.file_name, "test.rs");
        assert_eq!(deltas(&diff).len(), 1);
        assert_eq!(
            deltas(&diff)[0],
            DiffDelta {
                replacement_line_range: 2..3,
                insertion: "    println!(\"Hello, World!\");".to_string(),
            }
        );
    }

    #[test]
    fn test_v4a_with_change_context() {
        let hunks = vec![V4AHunk {
            change_context: vec!["impl MyStruct {".to_string()],
            pre_context: "    fn method1() {\n        // comment".to_string(),
            old: "        let x = 1;".to_string(),
            new: "        let x = 2;".to_string(),
            post_context: "    }\n}".to_string(),
        }];

        let file_content = "struct MyStruct {}\n\nimpl MyStruct {\n    fn method1() {\n        // comment\n        let x = 1;\n    }\n}";
        let diff = fuzzy_match_v4a_diffs("test.rs", &hunks, None, file_content);

        assert_eq!(deltas(&diff).len(), 1);
        assert_eq!(
            deltas(&diff)[0],
            DiffDelta {
                replacement_line_range: 6..7,
                insertion: "        let x = 2;".to_string(),
            }
        );
    }

    #[test]
    fn test_v4a_indentation_agnostic_match() {
        let hunks = vec![V4AHunk {
            change_context: vec![],
            pre_context: "def hello():".to_string(),
            old: "print(\"hello\")".to_string(),
            new: "    print(\"hello world\")".to_string(),
            post_context: "".to_string(),
        }];

        let file_content = "def hello():\n    print(\"hello\")";
        let diff = fuzzy_match_v4a_diffs("test.py", &hunks, None, file_content);

        assert_eq!(deltas(&diff).len(), 1);
        assert_eq!(
            deltas(&diff)[0],
            DiffDelta {
                replacement_line_range: 2..3,
                insertion: "    print(\"hello world\")".to_string(),
            }
        );
    }

    #[test]
    fn test_v4a_fuzzy_match() {
        let hunks = vec![V4AHunk {
            change_context: vec![],
            pre_context: "function greet() {".to_string(),
            old: "    console.log(\"helo\");".to_string(),
            new: "    console.log(\"hello world\");".to_string(),
            post_context: "}".to_string(),
        }];

        let file_content = "function greet() {\n    console.log(\"hello\");\n}";
        let diff = fuzzy_match_v4a_diffs("test.js", &hunks, None, file_content);

        assert_eq!(deltas(&diff).len(), 1);
        assert_eq!(deltas(&diff)[0].replacement_line_range, 2..3);
    }

    #[test]
    fn test_v4a_no_match() {
        let hunks = vec![V4AHunk {
            change_context: vec![],
            pre_context: "fn does_not_exist() {".to_string(),
            old: "    unrelated_code();".to_string(),
            new: "    new_code();".to_string(),
            post_context: "}".to_string(),
        }];

        let file_content = "fn main() {\n    println!(\"Hello\");\n}";
        let diff = fuzzy_match_v4a_diffs("test.rs", &hunks, None, file_content);

        assert!(deltas(&diff).is_empty());
        assert!(diff.failures.is_some());
        let failures = diff.failures.unwrap();
        assert_eq!(failures.fuzzy_match_failures, 1);
    }

    #[test]
    fn test_v4a_noop_diff() {
        let hunks = vec![V4AHunk {
            change_context: vec![],
            pre_context: "fn main() {".to_string(),
            old: "    println!(\"Hello\");".to_string(),
            new: "    println!(\"Hello\");".to_string(),
            post_context: "}".to_string(),
        }];

        let file_content = "fn main() {\n    println!(\"Hello\");\n}";
        let diff = fuzzy_match_v4a_diffs("test.rs", &hunks, None, file_content);

        assert!(deltas(&diff).is_empty());
        assert!(diff.failures.is_some());
        let failures = diff.failures.unwrap();
        assert_eq!(failures.noop_deltas, 1);
    }

    #[test]
    fn test_v4a_empty_context() {
        let hunks = vec![V4AHunk {
            change_context: vec![],
            pre_context: String::new(),
            old: "let x = 1;".to_string(),
            new: "let x = 2;".to_string(),
            post_context: String::new(),
        }];

        let file_content = "let x = 1;";
        let diff = fuzzy_match_v4a_diffs("test.rs", &hunks, None, file_content);

        assert_eq!(deltas(&diff).len(), 1);
        assert_eq!(
            deltas(&diff)[0],
            DiffDelta {
                replacement_line_range: 1..2,
                insertion: "let x = 2;".to_string(),
            }
        );
    }

    #[test]
    fn test_v4a_multiline_old_content() {
        let hunks = vec![V4AHunk {
            change_context: vec![],
            pre_context: "fn calculate() {".to_string(),
            old: "    let a = 1;\n    let b = 2;\n    let sum = a + b;".to_string(),
            new: "    let sum = 3;".to_string(),
            post_context: "    println!(\"{}\", sum);\n}".to_string(),
        }];

        let file_content = "fn calculate() {\n    let a = 1;\n    let b = 2;\n    let sum = a + b;\n    println!(\"{}\", sum);\n}";
        let diff = fuzzy_match_v4a_diffs("test.rs", &hunks, None, file_content);

        assert_eq!(deltas(&diff).len(), 1);
        assert_eq!(
            deltas(&diff)[0],
            DiffDelta {
                replacement_line_range: 2..5,
                insertion: "    let sum = 3;".to_string(),
            }
        );
    }

    #[test]
    fn test_v4a_multiple_hunks() {
        let hunks = vec![
            V4AHunk {
                change_context: vec![],
                pre_context: "fn first() {".to_string(),
                old: "    let x = 1;".to_string(),
                new: "    let x = 10;".to_string(),
                post_context: "}".to_string(),
            },
            V4AHunk {
                change_context: vec![],
                pre_context: "fn second() {".to_string(),
                old: "    let y = 2;".to_string(),
                new: "    let y = 20;".to_string(),
                post_context: "}".to_string(),
            },
        ];

        let file_content = "fn first() {\n    let x = 1;\n}\n\nfn second() {\n    let y = 2;\n}";
        let diff = fuzzy_match_v4a_diffs("test.rs", &hunks, None, file_content);

        assert_eq!(deltas(&diff).len(), 2);
        assert_eq!(deltas(&diff)[0].replacement_line_range, 2..3);
        assert_eq!(deltas(&diff)[0].insertion, "    let x = 10;");
        assert_eq!(deltas(&diff)[1].replacement_line_range, 6..7);
        assert_eq!(deltas(&diff)[1].insertion, "    let y = 20;");
    }

    #[test]
    fn test_v4a_add_line_with_change_context_no_old() {
        let hunks = vec![V4AHunk {
            change_context: vec!["class MyClass {".to_string()],
            pre_context: "".to_string(),
            old: "".to_string(),
            new: "    fn new_method() {\n        return 2;\n    }".to_string(),
            post_context: "    fn existing_method() {".to_string(),
        }];

        let file_content = "class MyClass {\n    fn existing_method() {\n        return 1;\n    }\n}";
        let diff = fuzzy_match_v4a_diffs("test.rs", &hunks, None, file_content);

        assert_eq!(deltas(&diff).len(), 1);
        assert_eq!(deltas(&diff)[0].replacement_line_range, 2..2);
        assert_eq!(
            deltas(&diff)[0].insertion,
            "    fn new_method() {\n        return 2;\n    }"
        );
    }

    #[test]
    fn test_v4a_add_line_at_start_of_file() {
        let hunks = vec![V4AHunk {
            change_context: vec![],
            pre_context: "".to_string(),
            old: "".to_string(),
            new: "// New header comment".to_string(),
            post_context: "fn main() {".to_string(),
        }];

        let file_content = "fn main() {\n    println!(\"Hello\");\n}";
        let diff = fuzzy_match_v4a_diffs("test.rs", &hunks, None, file_content);

        assert_eq!(deltas(&diff).len(), 1);
        assert_eq!(deltas(&diff)[0].replacement_line_range, 1..1);
        assert_eq!(deltas(&diff)[0].insertion, "// New header comment");
    }

    #[test]
    fn test_v4a_add_line_at_end_of_file() {
        let hunks = vec![V4AHunk {
            change_context: vec![],
            pre_context: "fn main() {\n    println!(\"Hello\");\n}".to_string(),
            old: "".to_string(),
            new: "\n// Footer comment".to_string(),
            post_context: "".to_string(),
        }];

        let file_content = "fn main() {\n    println!(\"Hello\");\n}";
        let diff = fuzzy_match_v4a_diffs("test.rs", &hunks, None, file_content);

        assert_eq!(deltas(&diff).len(), 1);
        assert_eq!(deltas(&diff)[0].replacement_line_range, 4..4);
        assert_eq!(deltas(&diff)[0].insertion, "\n// Footer comment");
    }

    #[test]
    fn test_partial_last_line_in_search_preserves_suffix() {
        let file_content = "func foo() {\nlet x = 1;\nlet x = 2;\n}";

        let diffs = [SearchAndReplace {
            search: "let x = 1;\nlet x".to_string(),
            replace: "let y = 1;\nlet x".to_string(),
        }];

        let (deltas, _failures) = fuzzy_match_file_diffs(&diffs, file_content);

        assert_eq!(deltas.len(), 1, "Expected one matched delta");
        assert_eq!(deltas[0].replacement_line_range, 2..4);
        assert_eq!(deltas[0].insertion, "let y = 1;\nlet x = 2;");

        let file_lines: Vec<&str> = file_content.lines().collect();
        let range = &deltas[0].replacement_line_range;
        let mut result = String::new();
        for line in &file_lines[..range.start - 1] {
            result.push_str(line);
            result.push('\n');
        }
        result.push_str(&deltas[0].insertion);
        result.push('\n');
        for line in &file_lines[range.end - 1..] {
            result.push_str(line);
            result.push('\n');
        }
        assert_eq!(result, "func foo() {\nlet y = 1;\nlet x = 2;\n}\n");
    }

    /// Search/replace pair is non-trivial, but applied to file content the
    /// effect is a no-op — skip it.
    #[test]
    fn test_replace_matches_file_content() {
        let diffs = [SearchAndReplace {
            search: "1|Hey, there".to_string(),
            replace: "Hi, there".to_string(),
        }];
        let (deltas, errors) = fuzzy_match_file_diffs(&diffs, "Hi, there\nGoodbye, world");
        assert!(deltas.is_empty());
        assert_eq!(errors.noop_deltas, 1);
    }

    #[test]
    fn test_search_range_greater_than_file_length() {
        // This should not panic.
        let r = match_diff(
            "hey\nthere",
            Some(14..15),
            &["hey", "there"],
            1f64,
            MakeExactMatch,
        );

        assert_eq!(r, Some(1..3));
    }

    #[test]
    fn test_custom_lines() {
        assert_eq!(lines("").collect_vec(), vec![""]);
        assert_eq!(lines("foobar").collect_vec(), vec!["foobar"]);
        assert_eq!(lines("foo\nbar").collect_vec(), vec!["foo", "bar"]);
        assert_eq!(lines("foo\nbar\n").collect_vec(), vec!["foo", "bar"]);
    }
}
