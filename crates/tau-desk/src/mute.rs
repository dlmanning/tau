use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Set of `SessionSeed::seed_from` refs the user has muted from the
/// Now-zone projection. Lives outside the card store: muting hides the
/// suggestion from the Now zone but does not touch the activity log.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SuggestionMutes {
    refs: HashSet<String>,
}

impl SuggestionMutes {
    pub fn new() -> Self {
        Self {
            refs: HashSet::new(),
        }
    }

    pub fn mute(&mut self, _seed_from: impl Into<String>) {
        todo!()
    }

    pub fn unmute(&mut self, _seed_from: &str) -> bool {
        todo!()
    }

    pub fn is_muted(&self, _seed_from: &str) -> bool {
        todo!()
    }

    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.refs.iter()
    }
}
