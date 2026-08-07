//! Physical column-family names for the append-only ledger.

/// Logical column families used by storage backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ColumnFamily {
    /// Append-only command envelopes keyed by commit version.
    ResolutionLedger,
}

impl ColumnFamily {
    /// Returns the stable backend column-family name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ResolutionLedger => "resolution_ledger",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_column_name_is_stable() {
        assert_eq!(
            ColumnFamily::ResolutionLedger.as_str(),
            "resolution_ledger"
        );
    }
}
