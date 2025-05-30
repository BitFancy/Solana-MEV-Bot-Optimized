// src/ai/agent.rs

/// Represents the current state of the market.
/// Placeholder struct for now.
#[derive(Debug, Clone)]
pub struct MarketState {
    pub volatility: f64,    // Market volatility indicator
    pub trend: f64,         // Market trend indicator (e.g., -1 for bearish, 1 for bullish, 0 for neutral)
    pub liquidity: f64,     // Market liquidity indicator
}

impl MarketState {
    /// Creates a new MarketState.
    pub fn new(volatility: f64, trend: f64, liquidity: f64) -> Self {
        MarketState {
            volatility,
            trend,
            liquidity,
        }
    }
}

use crate::ai::optimizer::{StrategyOptimizer, RLState, RLAction, discretize_trend, discretize_volatility}; // For RL consultation
use crate::backtester::config::AIAgentConfig; // To use the new config struct

/// The AIAgent struct, responsible for making trading decisions.
#[derive(Debug)]
pub struct AIAgent {
    market_state: MarketState,
    risk_tolerance: f64,
    min_profit_threshold: f64,
    learning_rate: f64,
    performance_history: Vec<f64>,
    rl_veto_threshold: f64, // New field for RL Veto
}

impl AIAgent {
    /// Creates a new AIAgent using a configuration struct.
    pub fn new(config: &AIAgentConfig) -> Self {
        AIAgent {
            market_state: MarketState::new(0.0, 0.0, 0.0), // Default market state
            risk_tolerance: config.risk_tolerance,
            min_profit_threshold: config.min_profit_threshold,
            learning_rate: config.initial_learning_rate,
            performance_history: Vec::new(),
            rl_veto_threshold: config.rl_veto_threshold,
        }
    }

    /// Updates the agent's view of the market state.
    ///
    /// # Arguments
    ///
    /// * `new_state` - The new market state to update to.
    pub fn update_market_state(&mut self, new_state: MarketState) {
        self.market_state = new_state;
    }

    /// Evaluates a potential trading opportunity.
    ///
    /// # Arguments
    /// * `path_id` - Identifier for the trading path/opportunity.
    /// * `potential_profit_raw` - The raw profit calculated from simulation.
    /// * `predicted_change_abs` - Absolute predicted price change from MarketPredictor.
    /// * `prediction_confidence` - Confidence from MarketPredictor for its prediction.
    /// * `external_risk_score` - Risk score from external factors (e.g., path complexity).
    /// * `strategy_optimizer` - Optional: For consulting RL Q-values.
    ///
    /// # Returns
    /// `true` if the opportunity is worth pursuing, `false` otherwise.
    pub fn evaluate_opportunity(
        &self,
        path_id: &str,
        potential_profit_raw: f64,
        _predicted_change_abs: f64, // Currently unused directly, but available for future logic
        prediction_confidence: f64,
        external_risk_score: f64,
        strategy_optimizer: Option<&StrategyOptimizer>, // Optional for RL consultation
    ) -> bool {
        // Adjusted Profit Expectation
        // Ensure prediction_confidence is not excessively high (e.g. >1) or low (e.g. <0)
        let clamped_prediction_confidence = prediction_confidence.max(0.0).min(1.0);
        let adjusted_expected_profit = potential_profit_raw * clamped_prediction_confidence;

        // Risk Assessment
        // Internal risk considers prediction uncertainty and market volatility
        let internal_risk_score = (1.0 - clamped_prediction_confidence) + self.market_state.volatility * 0.5;
        let total_risk_score = (internal_risk_score + external_risk_score).max(0.0); // Ensure risk is not negative

        // Dynamic Profit Threshold based on risk
        // The idea: higher risk or lower confidence demands higher potential profit.
        // self.risk_tolerance acts as a multiplier on how much risk increases the profit threshold.
        // A risk_tolerance of 0 means risk doesn't increase the threshold.
        // A risk_tolerance of 1 means a risk score of 0.1 increases required profit by 10% over base min_profit_threshold.
        let dynamic_min_profit = self.min_profit_threshold * (1.0 + total_risk_score * self.risk_tolerance);

        log::debug!(
            "Evaluating Path [{}]: RawProfit={:.2}, AdjProfit={:.2} (Conf={:.2}), TotalRisk={:.3}, DynMinProfit={:.2}",
            path_id, potential_profit_raw, adjusted_expected_profit, clamped_prediction_confidence, total_risk_score, dynamic_min_profit
        );

        if adjusted_expected_profit < dynamic_min_profit {
            log::info!("Path [{}] REJECTED: Adjusted profit {:.2} below dynamic threshold {:.2}", path_id, adjusted_expected_profit, dynamic_min_profit);
            return false;
        }

        // Optional: Consult RL Q-values for a veto
        if let Some(optimizer) = strategy_optimizer {
            // Use a neutral path performance for querying Q-value, as we are deciding *before* execution.
            // More advanced: path_performance could be path's historical avg profit if available to agent.
            let rl_opinion = self.consult_rl_opinion(path_id, &self.market_state, optimizer, 0);
            if let Some(q_value) = rl_opinion {
                log::debug!("Path [{}]: RL Q-value opinion: {:.4}", path_id, q_value);
                if q_value < self.rl_veto_threshold {
                    log::info!("Path [{}] VETOED by RL: Q-value {:.4} below threshold {:.4}", path_id, q_value, self.rl_veto_threshold);
                    return false;
                }
            }
        }

        log::info!("Path [{}] ACCEPTED: Adjusted profit {:.2} meets dynamic threshold {:.2}", path_id, adjusted_expected_profit, dynamic_min_profit);
        true
    }

    /// Consults the StrategyOptimizer's Q-table for an opinion on executing a path.
    ///
    /// # Arguments
    /// * `path_id` - The ID of the path to evaluate.
    /// * `market_state` - The current market state.
    /// * `strategy_optimizer` - A reference to the `StrategyOptimizer`.
    /// * `path_context_performance` - An i8 representing recent performance context for this path (e.g. 0 for neutral/pre-trade).
    ///
    /// # Returns
    /// `Some(q_value)` if the state and action are found in the Q-table, `None` otherwise.
    pub fn consult_rl_opinion(
        &self,
        path_id: &str,
        market_state: &MarketState,
        strategy_optimizer: &StrategyOptimizer,
        path_context_performance: i8,
    ) -> Option<f64> {
        let rl_state = RLState {
            market_trend: discretize_trend(market_state.trend),
            market_volatility: discretize_volatility(market_state.volatility),
            // For pre-trade consultation, path_performance might be neutral (0)
            // or based on a general historical performance of the path if available to agent.
            recent_path_performance: path_context_performance,
        };
        let action = RLAction::ExecutePath(path_id.to_string());

        strategy_optimizer.q_table.get(&rl_state)
            .and_then(|actions_map| actions_map.get(&action).cloned())
    }


    /// Records the outcome of a trade and updates performance history.
    ///
    /// # Arguments
    ///
    /// * `profit` - The profit (positive) or loss (negative) from the trade.
    pub fn record_trade_outcome(&mut self, profit: f64) {
        self.performance_history.push(profit);
        // Optional: Keep the history to a fixed size, e.g., last 100 trades
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
        // After recording, adapt the learning rate.
        self.adapt_learning_rate();
    }

    /// Adjusts the learning rate based on performance history.
    /// If average performance is improving, decrease learning rate (exploitation).
    /// If decreasing or stagnant, increase learning rate (exploration).
    pub fn adapt_learning_rate(&mut self) {
        let history_len = self.performance_history.len();
        const MIN_TRADES_FOR_ADAPTATION: usize = 10; // Ensure at least 10 trades
        const ADAPTATION_WINDOW: usize = 5; // Use 5 trades for each segment

        if history_len < MIN_TRADES_FOR_ADAPTATION {
            log::debug!("Learning rate adaptation skipped: Not enough trade history ({} < {}).", history_len, MIN_TRADES_FOR_ADAPTATION);
            return;
        }

        // Compare average of the most recent `ADAPTATION_WINDOW` trades
        // with the `ADAPTATION_WINDOW` trades before that.
        let current_segment_avg: f64 = self.performance_history[history_len - ADAPTATION_WINDOW..].iter().sum::<f64>() / ADAPTATION_WINDOW as f64;
        let previous_segment_avg: f64 = self.performance_history[history_len - 2 * ADAPTATION_WINDOW..history_len - ADAPTATION_WINDOW].iter().sum::<f64>() / ADAPTATION_WINDOW as f64;

        let old_lr = self.learning_rate;
        if current_segment_avg > previous_segment_avg {
            // Performance is improving, exploit current strategy (decrease learning rate)
            self.learning_rate *= 0.95; // Decrease by 5%
        } else if current_segment_avg < previous_segment_avg {
            // Performance is decreasing, explore new strategies (increase learning rate)
            self.learning_rate *= 1.05; // Increase by 5%
        }
        // If performance is stagnant (averages are equal), no change to learning rate.

        // Clamp learning rate to a reasonable range, e.g., [0.01, 0.5]
        self.learning_rate = self.learning_rate.max(0.01).min(0.5);
        if (old_lr - self.learning_rate).abs() > 1e-6 { // Log if changed
             log::info!("AIAgent learning rate adapted from {:.4} to {:.4} (HistoryLen: {}, PrevAvg: {:.2}, CurrAvg: {:.2})",
                old_lr, self.learning_rate, history_len, previous_segment_avg, current_segment_avg);
        }
    }

    /// Returns the agent's current minimum profit threshold.
    pub fn get_current_profit_threshold(&self) -> f64 {
        self.min_profit_threshold
    }

    /// Returns the agent's risk tolerance.
    pub fn get_risk_tolerance(&self) -> f64 {
        self.risk_tolerance
    }

    /// Returns the agent's current learning rate.
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the agent's current market state.
    pub fn get_market_state(&self) -> &MarketState {
        &self.market_state
    }
}

// Basic tests for AIAgent
#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap; // For mocking QTable in tests
    use crate::ai::optimizer::RLState; // Ensure RLState is accessible for tests

    fn get_default_config() -> AIAgentConfig {
        AIAgentConfig::default()
    }

    #[test]
    fn test_agent_creation_with_config() {
        let config = get_default_config();
        let agent = AIAgent::new(&config);
        assert_eq!(agent.get_risk_tolerance(), config.risk_tolerance);
        assert_eq!(agent.get_current_profit_threshold(), config.min_profit_threshold);
        assert_eq!(agent.get_learning_rate(), config.initial_learning_rate);
        assert_eq!(agent.rl_veto_threshold, config.rl_veto_threshold);
        assert!(agent.performance_history.is_empty());
    }

    #[test]
    fn test_update_market_state() {
        let mut agent = AIAgent::new(&get_default_config());
        let initial_market_state = agent.get_market_state().clone();
        assert_eq!(initial_market_state.volatility, 0.0); // Default

        let new_state = MarketState::new(0.8, 1.0, 0.9);
        agent.update_market_state(new_state.clone());

        let updated_market_state = agent.get_market_state();
        assert_eq!(updated_market_state.volatility, 0.8);
        assert_eq!(updated_market_state.trend, 1.0);
        assert_eq!(updated_market_state.liquidity, 0.9);
    }

    #[test]
    fn test_evaluate_opportunity_enhanced() {
        let mut config = get_default_config();
        config.min_profit_threshold = 100.0; // Base profit needed
        config.risk_tolerance = 0.5;         // How much risk scales required profit
        let agent = AIAgent::new(&config);

        // Scenario 1: Good profit, high confidence, low risk
        // AdjustedProfit = 200 * 0.9 = 180
        // InternalRisk = (1-0.9) + 0.1*0.5 = 0.1 + 0.05 = 0.15
        // TotalRisk = 0.15 + 0.1 = 0.25
        // DynMinProfit = 100 * (1 + 0.25 * 0.5) = 100 * (1 + 0.125) = 112.5
        // 180 >= 112.5 -> true
        let mut market_state_low_vol = agent.market_state.clone();
        market_state_low_vol.volatility = 0.1;
        let mut agent_low_vol = AIAgent::new(&config);
        agent_low_vol.market_state = market_state_low_vol;
        assert!(agent_low_vol.evaluate_opportunity("path1", 200.0, 0.02, 0.9, 0.1, None));

        // Scenario 2: Profit below dynamic threshold due to low confidence
        // AdjustedProfit = 150 * 0.5 = 75
        // InternalRisk = (1-0.5) + 0.1*0.5 = 0.5 + 0.05 = 0.55
        // TotalRisk = 0.55 + 0.1 = 0.65
        // DynMinProfit = 100 * (1 + 0.65 * 0.5) = 100 * (1 + 0.325) = 132.5
        // 75 < 132.5 -> false
        assert!(!agent_low_vol.evaluate_opportunity("path2", 150.0, 0.015, 0.5, 0.1, None));

        // Scenario 3: Profit below dynamic threshold due to high external risk
        // AdjustedProfit = 200 * 0.9 = 180
        // InternalRisk = 0.15 (as scenario 1)
        // TotalRisk = 0.15 + 0.8 = 0.95
        // DynMinProfit = 100 * (1 + 0.95 * 0.5) = 100 * (1 + 0.475) = 147.5
        // 180 >= 147.5 -> true (oops, let's adjust external risk to make it fail)
        // TotalRisk = 0.15 + 1.5 = 1.65
        // DynMinProfit = 100 * (1 + 1.65 * 0.5) = 100 * (1 + 0.825) = 182.5
        // 180 < 182.5 -> false
        assert!(!agent_low_vol.evaluate_opportunity("path3", 200.0, 0.02, 0.9, 1.5, None));

        // Scenario 4: Barely meets threshold
        // AdjustedProfit = 150 * 0.8 = 120
        // InternalRisk = (1-0.8) + 0.1*0.5 = 0.2 + 0.05 = 0.25
        // TotalRisk = 0.25 + 0.2 = 0.45
        // DynMinProfit = 100 * (1 + 0.45 * 0.5) = 100 * (1 + 0.225) = 122.5
        // 120 < 122.5 -> false. Let's make potential_profit_raw 160. Adjusted = 160*0.8 = 128.
        // 128 > 122.5 -> true
        assert!(agent_low_vol.evaluate_opportunity("path4", 160.0, 0.016, 0.8, 0.2, None));
    }

    #[test]
    fn test_evaluate_opportunity_with_rl_veto() {
        let mut config = get_default_config();
        config.min_profit_threshold = 100.0;
        config.rl_veto_threshold = -0.1; // Veto if Q-value is less than -0.1
        let agent = AIAgent::new(&config);

        let mut optimizer = StrategyOptimizer::new(); // RL Q-Table is in optimizer
        let state = RLState { market_trend: 0, market_volatility: 0, recent_path_performance: 0 };
        let action = RLAction::ExecutePath("path_veto".to_string());

        // Insert a Q-value that should cause a veto
        let mut actions_map = HashMap::new();
        actions_map.insert(action.clone(), -0.5); // This Q-value is < rl_veto_threshold
        optimizer.q_table.insert(state, actions_map);

        // This opportunity would normally pass without RL consultation
        let should_pass_normally = agent.evaluate_opportunity("path_veto", 200.0, 0.02, 0.9, 0.1, None);
        assert!(should_pass_normally, "Opportunity should pass basic checks.");

        // Now, evaluate with RL consultation
        let decision_with_rl = agent.evaluate_opportunity("path_veto", 200.0, 0.02, 0.9, 0.1, Some(&optimizer));
        assert!(!decision_with_rl, "Opportunity should be vetoed by RL due to low Q-value.");

        // Test with Q-value that does not veto
        let mut optimizer_no_veto = StrategyOptimizer::new();
        let mut actions_map_no_veto = HashMap::new();
        actions_map_no_veto.insert(action, 0.2); // This Q-value is >= rl_veto_threshold
        optimizer_no_veto.q_table.insert(state, actions_map_no_veto);
        let decision_no_veto = agent.evaluate_opportunity("path_veto", 200.0, 0.02, 0.9, 0.1, Some(&optimizer_no_veto));
        assert!(decision_no_veto, "Opportunity should not be vetoed by RL with positive Q-value.");
    }


    #[test]
    fn test_record_trade_outcome_and_history_size() {
        let mut agent = AIAgent::new(&get_default_config());
        for i in 0..105 { // Add 105 trades
            agent.record_trade_outcome(i as f64 * 0.01);
        }
        assert_eq!(agent.performance_history.len(), 100); // Max history size is 100
        assert_eq!(agent.performance_history[0], 5.0 * 0.01); // First element after trimming (oldest)
        assert_eq!(agent.performance_history[99], 104.0 * 0.01); // Last element (newest)
    }

    #[test]
    fn test_adapt_learning_rate_logic() {
        let mut agent = AIAgent::new(&get_default_config());
        agent.learning_rate = 0.1; // Known starting rate

        // Scenario 1: Performance decreasing -> increase LR
        agent.performance_history.clear();
        for i in (0..10).rev() { agent.performance_history.push(i as f64 * 0.1); } // Decreasing profits
        // last 5 avg: (0+0.1+0.2+0.3+0.4)/5 = 0.2
        // prev 5 avg: (0.5+0.6+0.7+0.8+0.9)/5 = 0.7
        agent.adapt_learning_rate();
        assert!((agent.learning_rate - 0.1 * 1.05).abs() < 1e-6); // Increased by 5%

        // Scenario 2: Performance increasing -> decrease LR
        let current_lr = agent.learning_rate;
        agent.performance_history.clear();
        for i in 0..10 { agent.performance_history.push(i as f64 * 0.1); } // Increasing profits
        // last 5 avg: (0.5+0.6+0.7+0.8+0.9)/5 = 0.7
        // prev 5 avg: (0+0.1+0.2+0.3+0.4)/5 = 0.2
        agent.adapt_learning_rate();
        assert!((agent.learning_rate - current_lr * 0.95).abs() < 1e-6); // Decreased by 5%

        // Scenario 3: Not enough data -> no change
        let current_lr_no_change = agent.learning_rate;
        agent.performance_history.clear();
        for i in 0..9 { agent.performance_history.push(i as f64 * 0.1); } // Only 9 trades
        agent.adapt_learning_rate();
        assert_eq!(agent.learning_rate, current_lr_no_change);

        // Scenario 4: Stagnant performance -> no change
        agent.performance_history.clear();
        for _ in 0..10 { agent.performance_history.push(0.5); } // All profits are 0.5
        let stagnant_lr_before = agent.learning_rate;
        agent.adapt_learning_rate();
        assert_eq!(agent.learning_rate, stagnant_lr_before);

        // Scenario 5: Clamping
        agent.learning_rate = 0.001; // Set very low
        agent.performance_history.clear();
        for i in 0..10 { agent.performance_history.push(i as f64 * 0.1); } // Increasing profits (attempts to decrease LR)
        agent.adapt_learning_rate(); // Will try to set to 0.001 * 0.95 = 0.00095
        assert_eq!(agent.learning_rate, 0.01); // Should be clamped to min 0.01

        agent.learning_rate = 0.6; // Set very high
        agent.performance_history.clear();
        for i in (0..10).rev() { agent.performance_history.push(i as f64 * 0.1); } // Decreasing profits (attempts to increase LR)
        agent.adapt_learning_rate(); // Will try to set to 0.6 * 1.05 = 0.63
        assert_eq!(agent.learning_rate, 0.5); // Should be clamped to max 0.5
    }
}
