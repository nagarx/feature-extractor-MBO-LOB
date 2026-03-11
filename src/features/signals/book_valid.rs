//! Order book validity checks.

use mbo_lob_reconstructor::LobState;

/// Check if the order book is valid for trading.
///
/// A valid book has:
/// 1. Both bid and ask prices present
/// 2. Bid < Ask (not crossed)
/// 3. Positive spread
///
/// # Arguments
///
/// * `best_bid` - Best bid price (None if empty)
/// * `best_ask` - Best ask price (None if empty)
///
/// # Returns
///
/// `true` if book is valid, `false` otherwise.
#[inline]
pub fn is_book_valid(best_bid: Option<i64>, best_ask: Option<i64>) -> bool {
    match (best_bid, best_ask) {
        (Some(bid), Some(ask)) => bid < ask,
        _ => false,
    }
}

/// Check if book is valid from LobState.
#[inline]
pub fn is_book_valid_from_lob(lob: &LobState) -> bool {
    is_book_valid(lob.best_bid, lob.best_ask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_book_valid_normal_spread() {
        assert!(is_book_valid(Some(100_000_000), Some(100_010_000)));
    }

    #[test]
    fn test_is_book_valid_crossed() {
        assert!(!is_book_valid(Some(100_010_000), Some(100_000_000)));
    }

    #[test]
    fn test_is_book_valid_locked() {
        assert!(!is_book_valid(Some(100_000_000), Some(100_000_000)));
    }

    #[test]
    fn test_is_book_valid_empty_bid() {
        assert!(!is_book_valid(None, Some(100_000_000)));
    }

    #[test]
    fn test_is_book_valid_empty_ask() {
        assert!(!is_book_valid(Some(100_000_000), None));
    }

    #[test]
    fn test_is_book_valid_both_empty() {
        assert!(!is_book_valid(None, None));
    }

    #[test]
    fn test_is_book_valid_wide_spread() {
        assert!(is_book_valid(Some(100_000_000), Some(110_000_000)));
    }

    #[test]
    fn test_is_book_valid_one_tick_spread() {
        // Minimum valid spread (1 tick = $0.01 = 10,000 in fixed-point)
        assert!(is_book_valid(Some(100_000_000), Some(100_010_000)));
    }
}
