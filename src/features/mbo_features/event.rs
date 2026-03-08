//! MBO event representation for feature extraction.
//!
//! Lightweight event type that bridges the `mbo_lob_reconstructor::MboMessage`
//! format to the feature extractor's rolling window and aggregation pipeline.

use mbo_lob_reconstructor::{Action, MboMessage, Side};

/// MBO event for aggregation.
///
/// Lightweight representation of an order book event for efficient storage
/// in rolling windows.
#[derive(Debug, Clone)]
pub struct MboEvent {
    /// Event timestamp (nanoseconds since epoch)
    pub timestamp: u64,

    /// Order action (Add, Modify, Cancel, Trade)
    pub action: Action,

    /// Order side (Bid or Ask)
    pub side: Side,

    /// Order price (fixed-point: divide by 1e9 for dollars)
    pub price: i64,

    /// Order size (shares)
    pub size: u32,

    /// Unique order identifier
    pub order_id: u64,
}

impl MboEvent {
    /// Create a new MBO event from message components.
    #[inline]
    pub fn new(
        timestamp: u64,
        action: Action,
        side: Side,
        price: i64,
        size: u32,
        order_id: u64,
    ) -> Self {
        Self {
            timestamp,
            action,
            side,
            price,
            size,
            order_id,
        }
    }

    /// Create an MBO event from an MboMessage.
    ///
    /// Convenience method for converting from the reconstructor's
    /// message format to the feature extractor's event format.
    #[inline]
    pub fn from_mbo_message(msg: &MboMessage) -> Self {
        Self {
            timestamp: msg.timestamp.unwrap_or(0) as u64,
            action: msg.action,
            side: msg.side,
            price: msg.price,
            size: msg.size,
            order_id: msg.order_id,
        }
    }

    /// Convert to an MboMessage for use with the reconstructor's trackers.
    ///
    /// This enables integration with `QueuePositionTracker` and other
    /// reconstructor components that require `MboMessage` input.
    #[inline]
    pub fn to_mbo_message(&self) -> MboMessage {
        MboMessage {
            order_id: self.order_id,
            action: self.action,
            side: self.side,
            price: self.price,
            size: self.size,
            timestamp: Some(self.timestamp as i64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbo_event_creation() {
        let event = MboEvent::new(
            1000000000,
            Action::Add,
            Side::Bid,
            100_000_000_000,
            100,
            12345,
        );

        assert_eq!(event.timestamp, 1000000000);
        assert_eq!(event.size, 100);
        assert_eq!(event.order_id, 12345);
    }
}
