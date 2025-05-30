// src/ai/optimizer.rs

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use rand::Rng; // For epsilon-greedy exploration

// --- RL Core Components ---

/// Represents a discretized state for the Q-learning agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RLState {
    pub market_trend: i8,          // -1 (down), 0 (neutral), 1 (up)
    pub market_volatility: i8,     // 0 (low), 1 (medium), 2 (high)
    pub recent_path_performance: i8, // -1 (loss), 0 (neutral/no trade), 1 (profit) for a specific path context
}

// Manual implementation of Hash for RLState because f64 fields (if any) are not directly hashable,
// but here we only have i8 fields which are.
impl Hash for RLState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.market_trend.hash(state);
        self.market_volatility.hash(state);
        self.recent_path_performance.hash(state);
    }
}

/// Represents an action the Q-learning agent can take.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RLAction {
    ExecutePath(String), // String is path_id
    AvoidPath(String),   // String is path_id
    Hold,                // Generic hold action
}

/// Q-Table: Maps (State, Action) pairs to Q-values (f64).
pub type QTable = HashMap<RLState, HashMap<RLAction, f64>>;


// --- Original Structs (MarketStateSnapshot, TradingPath, TradeOutcome) ---
// These are kept from the previous implementation.

/// A snapshot of market conditions at a specific time.
/// Simplified version for logging and strategy optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct MarketStateSnapshot {
    pub volatility: f64,
    pub trend: f64, // -1 for down, 0 for neutral, 1 for up
}

impl MarketStateSnapshot {
    pub fn new(volatility: f64, trend: f64) -> Self {
        Self { volatility, trend }
    }
}

/// Represents a potential trading path or strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct TradingPath {
    pub id: String,
    pub steps: Vec<String>,
    pub expected_profit: f64, // This might be deprecated if RL takes over expected value estimation
    pub success_rate: f64,    // Similarly, this might be an output of RL or other analysis
    pub total_profit_generated: f64,
    pub times_traded: u32,
}

impl TradingPath {
    pub fn new(id: String, steps: Vec<String>, expected_profit: f64, success_rate: f64) -> Self {
        TradingPath {
            id,
            steps,
            expected_profit,
            success_rate,
            total_profit_generated: 0.0,
            times_traded: 0,
        }
    }

    pub fn average_profit(&self) -> f64 {
        if self.times_traded == 0 { 0.0 } else { self.total_profit_generated / self.times_traded as f64 }
    }
}

/// Represents the outcome of a single trade for learning and optimization.
#[derive(Debug, Clone)]
pub struct TradeOutcome {
    pub path_id: String,
    pub profit: f64,
    pub market_conditions_snapshot: MarketStateSnapshot,
    pub timestamp: u64,
}

impl TradeOutcome {
    pub fn new(path_id: String, profit: f64, market_conditions_snapshot: MarketStateSnapshot, timestamp: u64) -> Self {
        Self { path_id, profit, market_conditions_snapshot, timestamp }
    }
}


/// The StrategyOptimizer struct, responsible for refining trading strategies using Q-learning.
#[derive(Debug)]
pub struct StrategyOptimizer {
    performance_metrics: HashMap<String, f64>,
    pub best_paths: HashMap<String, TradingPath>, // Made public for easier access in strategy for now
    learning_model: Option<String>, // Placeholder for saving/loading RL model info
    trade_history_for_rl: Vec<TradeOutcome>,

    // Q-Learning specific fields
    q_table: QTable,
    learning_rate_rl: f64,      // alpha
    discount_factor_rl: f64,    // gamma
    exploration_rate_rl: f64,   // epsilon
}

// --- Helper functions for RL ---
pub fn discretize_volatility(volatility: f64) -> i8 {
    if volatility < 0.02 { // Example: Very low volatility (adjust thresholds as needed)
        0 // Low
    } else if volatility < 0.05 { // Example: Moderate volatility
        1 // Medium
    } else {
        2 // High
    }
}

pub fn discretize_trend(trend: f64) -> i8 {
    if trend < -0.1 { // Strong downtrend
        -1 // Down
    } else if trend > 0.1 { // Strong uptrend
        1 // Up
    } else {
        0 // Neutral / Weak trend
    }
}


impl StrategyOptimizer {
    /// Creates a new StrategyOptimizer with Q-learning components.
    pub fn new() -> Self {
        StrategyOptimizer {
            performance_metrics: HashMap::new(),
            best_paths: HashMap::new(),
            learning_model: None,
            trade_history_for_rl: Vec::new(),
            q_table: QTable::new(),
            learning_rate_rl: 0.1,    // Alpha: learning rate for Q-value updates
            discount_factor_rl: 0.9,  // Gamma: importance of future rewards
            exploration_rate_rl: 0.1, // Epsilon: for exploration vs exploitation
        }
    }

    // --- Q-Learning Core Methods ---

    /// Chooses an action based on the current state using an epsilon-greedy strategy.
    pub fn choose_action(&self, state: &RLState, available_path_ids: &Vec<String>) -> RLAction {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.exploration_rate_rl {
            // Exploration: Choose a random action
            if available_path_ids.is_empty() { // Should not happen if called correctly
                log::warn!("RL Choose Action (Explore): No available paths to choose from, defaulting to Hold.");
                return RLAction::Hold;
            }
            let random_path_index = rng.gen_range(0..available_path_ids.len());
            let path_id = available_path_ids[random_path_index].clone();
            if rng.gen_bool(0.7) { // 70% chance to explore executing a path
                log::debug!("RL Action (Explore): ExecutePath({})", path_id);
                RLAction::ExecutePath(path_id)
            } else { // 30% chance to explore avoiding
                log::debug!("RL Action (Explore): AvoidPath({})", path_id);
                RLAction::AvoidPath(path_id)
            }
        } else {
            // Exploitation: Choose the best action from Q-table
            if let Some(actions_for_state) = self.q_table.get(state) {
                let mut best_action = RLAction::Hold;
                let mut max_q_value = f64::NEG_INFINITY;

                // Iterate through available paths to form potential ExecutePath actions
                for path_id in available_path_ids {
                    let action_execute = RLAction::ExecutePath(path_id.clone());
                    let q_value_execute = actions_for_state.get(&action_execute).cloned().unwrap_or(0.0);
                    if q_value_execute > max_q_value {
                        max_q_value = q_value_execute;
                        best_action = action_execute;
                    }
                    // Optionally, consider AvoidPath actions if they are explicitly stored and valued
                    // let action_avoid = RLAction::AvoidPath(path_id.clone());
                    // let q_value_avoid = actions_for_state.get(&action_avoid).cloned().unwrap_or(0.0); // Default Q for Avoid might be different
                    // if q_value_avoid > max_q_value { ... }
                }

                // If no positive Q-value action is found among available paths, default to Hold or Avoid
                if max_q_value == f64::NEG_INFINITY || max_q_value <= 0.0 { // max_q_value can be 0 if all are unknown or non-positive
                    // Potentially select a random available path to avoid if no good execute option
                    if !available_path_ids.is_empty() && max_q_value <=0.0 { // If best is not profitable
                        // best_action = RLAction::AvoidPath(available_path_ids[rng.gen_range(0..available_path_ids.len())].clone());
                        // log::debug!("RL Action (Exploit/Default to Avoid): {:?} due to non-positive Q-values for Execute.", best_action);
                        // For now, if no good ExecutePath, stick to Hold or the best of non-positives.
                        // This part can be refined: e.g. if all Q-values for Execute are <=0, should it pick Avoid?
                        // Current logic picks ExecutePath with highest Q, even if that Q is 0 or negative.
                    } else if max_q_value == f64::NEG_INFINITY { // No actions for state or no available paths led to q-value update
                         log::debug!("RL Action (Exploit/Default): No Q-values for state {:?} or no valid paths, defaulting to Hold.", state);
                         best_action = RLAction::Hold;
                    }
                }
                log::debug!("RL Action (Exploit): {:?} with Q-value: {:.4}", best_action, max_q_value);
                best_action
            } else {
                log::debug!("RL Action (Exploit/Default): No Q-values map for state {:?}, defaulting to Hold.", state);
                RLAction::Hold
            }
        }
    }

    /// Updates the Q-table based on the Q-learning rule.
    /// Simplified: next_max_q_value is assumed to be 0 for this iteration (no lookahead to s').
    pub fn update_q_table(&mut self, state: &RLState, action: &RLAction, reward: f64, _next_state: &RLState) {
        let old_q_value = self.q_table.entry(*state).or_insert_with(HashMap::new)
                                   .get(action).cloned().unwrap_or(0.0);

        // Simplified Q-learning: Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
        // This is because we assume max_a' Q(s',a') = 0 (no future state evaluation yet)
        let next_max_q_value = 0.0; // Simplified for now

        let new_q_value = old_q_value + self.learning_rate_rl * (reward + self.discount_factor_rl * next_max_q_value - old_q_value);

        self.q_table.entry(*state).or_insert_with(HashMap::new).insert(action.clone(), new_q_value);
        log::info!(
            "Q-Table Update: State={:?}, Action={:?}, Reward={:.2}, OldQ={:.3}, NewQ={:.3}",
            state, action, reward, old_q_value, new_q_value
        );
    }

    // --- Modified existing methods ---

    /// Records a trade's outcome, updates path stats, overall metrics, and the Q-table.
    ///
    /// # Arguments
    ///
    /// * `trade_outcome` - The outcome of the trade to record.
    pub fn track_trade(&mut self, trade_outcome: TradeOutcome) {
        log::info!(
            "Tracking trade for path_id: {}, Profit: {:.4}",
            trade_outcome.path_id,
            trade_outcome.profit
        );

        // Update standard path statistics
        if let Some(path) = self.best_paths.get_mut(&trade_outcome.path_id) {
            path.total_profit_generated += trade_outcome.profit;
            path.times_traded += 1;
            if trade_outcome.profit > 0.0 {
                path.success_rate = (path.success_rate * (path.times_traded -1) as f64 + 1.0) / path.times_traded as f64;
            } else {
                path.success_rate = (path.success_rate * (path.times_traded -1) as f64) / path.times_traded as f64;
            }
        } else {
            log::warn!("Attempted to track trade for unknown path_id: {}", trade_outcome.path_id);
        }

        // Q-Learning Update
        let reward = trade_outcome.profit;
        let current_market_trend = discretize_trend(trade_outcome.market_conditions_snapshot.trend);
        let current_market_volatility = discretize_volatility(trade_outcome.market_conditions_snapshot.volatility);
        let current_path_performance = if reward > 0.0 { 1 } else if reward < 0.0 { -1 } else { 0 };

        let state = RLState {
            market_trend: current_market_trend,
            market_volatility: current_market_volatility,
            recent_path_performance: current_path_performance, // This is context for THIS path, if it were just traded.
                                                              // For a general state, this could be overall agent perf or avg path perf.
        };

        // Action is assumed to be ExecutePath for the traded path.
        let action = RLAction::ExecutePath(trade_outcome.path_id.clone());

        // Simplified next_state: for this iteration, assume current state is also next state for Q-value estimation,
        // or a generic "terminal" state if that makes more sense. Here, using current state.
        // The Q-learning rule uses max_a' Q(s',a'), which is simplified to 0 for now.
        let dummy_next_state = state; // Simplified: Q-value of next state's best action is 0.
                                     // A more advanced version would predict or use the actual next state.

        self.update_q_table(&state, &action, reward, &dummy_next_state);

        self.trade_history_for_rl.push(trade_outcome); // Keep history after RL update
        self.update_performance_metrics();
    }

    /// Iterates through known paths and logs RL's preferred action based on a hypothetical current market state.
    /// NOTE: `current_market_snapshot` needs to be passed in or fetched. For now, uses last known from history if available.
    pub fn optimize_strategies(&mut self, current_market_snapshot_option: Option<MarketStateSnapshot>) {
        log::info!("StrategyOptimizer: optimize_strategies called. Evaluating paths with RL preferences...");

        let market_snapshot_to_use = current_market_snapshot_option.or_else(|| {
            self.trade_history_for_rl.last().map(|t| t.market_conditions_snapshot.clone())
        });

        if market_snapshot_to_use.is_none() {
            log::warn!("Optimize Strategies: No current or historical market snapshot available to form RL state. Skipping RL preference logging.");
            return;
        }
        let market_snapshot = market_snapshot_to_use.unwrap();

        let state_market_trend = discretize_trend(market_snapshot.trend);
        let state_market_volatility = discretize_volatility(market_snapshot.volatility);

        let available_path_ids: Vec<String> = self.best_paths.keys().cloned().collect();
        if available_path_ids.is_empty() {
            log::info!("Optimize Strategies: No paths available to evaluate.");
            return;
        }

        for path_id in &available_path_ids {
            // For path-specific part of state, we might use its historical average performance
            let path_avg_profit = self.best_paths.get(path_id).map_or(0.0, |p| p.average_profit());
            let state_path_performance = if path_avg_profit > 0.0 { 1 } else if path_avg_profit < 0.0 { -1 } else { 0 };

            let current_state_for_path = RLState {
                market_trend: state_market_trend,
                market_volatility: state_market_volatility,
                recent_path_performance: state_path_performance,
            };

            // Get available actions (all known paths for Execute/Avoid)
            let rl_action = self.choose_action(&current_state_for_path, &available_path_ids); // Pass all paths for action choice context
            log::info!(
                "RL Preference for path {}: Action: {:?} (based on state: {:?})",
                path_id, rl_action, current_state_for_path
            );
            // In future, this chosen action could influence path.expected_profit or a new 'q_score' field.
        }
    }

    /// Returns the `top_n` best-performing trading paths, sorted by average profit.
    /// (RL preferences not yet integrated into sorting here).
        // based on their historical average profit.
        // This is already implicitly done by `track_trade` updating `total_profit_generated` and `times_traded`,
        // and `average_profit()` method on TradingPath.
        // The main task here would be to potentially discover new paths or prune bad ones.

        // For now, let's just ensure paths are sorted by some metric if needed for `get_best_paths`.
        // `get_best_paths` handles the sorting.
        // A more advanced version might:
        // - Analyze `trade_history_for_rl` in conjunction with `market_conditions_snapshot`.
        // - Adjust path parameters or discover new path variations.
        // - Prune paths that consistently underperform.
        // - Update `learning_model`.
        // println!("Placeholder: optimize_strategies called. In a real system, RL would occur here."); // Replaced by log
    }

    /// Returns the `top_n` best-performing trading paths, sorted by average profit.
    ///
    /// # Arguments
    ///
    /// * `top_n` - The number of best paths to return.
    ///
    /// # Returns
    ///
    /// A vector of `TradingPath` structs.
    pub fn get_best_paths(&self, top_n: usize) -> Vec<TradingPath> {
        log::info!("StrategyOptimizer: get_best_paths called for top {} paths.", top_n);
        let mut paths_vec: Vec<TradingPath> = self.best_paths.values().cloned().collect();

        // Sort by average profit in descending order.
        // Paths with more trades and decent profit might be preferred over high profit with few trades.
        // For now, simple average profit.
        paths_vec.sort_by(|a, b| b.average_profit().partial_cmp(&a.average_profit()).unwrap_or(std::cmp::Ordering::Equal));

        paths_vec.into_iter().take(top_n).collect()
    }

    /// Adds or updates a trading path in the optimizer.
    /// Useful for initially populating or dynamically adding strategies.
    pub fn add_or_update_path(&mut self, path: TradingPath) {
        self.best_paths.insert(path.id.clone(), path);
    }


    /// Recalculates overall performance metrics based on `trade_history_for_rl`.
    fn update_performance_metrics(&mut self) {
        let total_trades = self.trade_history_for_rl.len();
        if total_trades == 0 {
            if !self.performance_metrics.is_empty() { // Log only if clearing existing metrics
                log::info!("StrategyOptimizer: update_performance_metrics - No trades in history, clearing metrics.");
            }
            self.performance_metrics.clear();
            return;
        }

        let total_profit: f64 = self.trade_history_for_rl.iter().map(|t| t.profit).sum();
        let winning_trades = self.trade_history_for_rl.iter().filter(|t| t.profit > 0.0).count();

        let average_profit = total_profit / total_trades as f64;
        let win_rate = winning_trades as f64 / total_trades as f64;

        self.performance_metrics.insert("total_trades".to_string(), total_trades as f64);
        self.performance_metrics.insert("total_profit".to_string(), total_profit);
        self.performance_metrics.insert("average_profit_per_trade".to_string(), average_profit);
        self.performance_metrics.insert("win_rate".to_string(), win_rate);

        log::info!(
            "StrategyOptimizer: update_performance_metrics - Total Trades: {}, Total Profit: {:.4}, Avg Profit: {:.4}, Win Rate: {:.2}",
            total_trades, total_profit, average_profit, win_rate
        );
        // Other metrics like Sharpe ratio, max drawdown, etc., could be added here.
    }

    /// Returns the current performance metrics.
    pub fn get_performance_metrics(&self) -> &HashMap<String, f64> {
        &self.performance_metrics
    }


    /// Placeholder for adapting strategies based on market conditions.
    /// Suggests paths or adjusts parameters. For now, this is a conceptual placeholder.
    ///
    /// # Arguments
    ///
    /// * `market_state` - The current market state snapshot.
    pub fn adapt_strategy_based_on_market(&mut self, market_state: &MarketStateSnapshot) {
        // Naive example: If market is highly volatile, it might prefer paths with historically
        // higher success rates, even if their average profit is slightly lower.
        // Or, it might adjust risk parameters for selected paths.
        log::info!("StrategyOptimizer: adapt_strategy_based_on_market called with market state: Volatility={:.4}, Trend={:.1}",
            market_state.volatility, market_state.trend
        );
        if market_state.volatility > 0.7 { // Arbitrary threshold for high volatility
            log::info!("StrategyOptimizer: High market volatility ({:.4}) detected. Suggests favoring high success rate paths.", market_state.volatility);
            // Actual logic would involve re-sorting or filtering `self.best_paths`
            // or communicating adjustments to the AIAgent.
        }
    }
}

// Unit tests for StrategyOptimizer
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_market_snapshot() -> MarketStateSnapshot {
        MarketStateSnapshot::new(0.5, 0.5) // Moderate volatility, slight uptrend
    }

    fn sample_path1() -> TradingPath {
        TradingPath::new("PATH001".to_string(), vec!["BUY BTC".to_string(), "SELL BTC".to_string()], 0.05, 0.7)
    }

    fn sample_path2() -> TradingPath {
         TradingPath::new("PATH002".to_string(), vec!["BUY ETH".to_string(), "SELL ETH".to_string()], 0.08, 0.6)
    }

    #[test]
    fn test_optimizer_creation_with_rl() {
        let optimizer = StrategyOptimizer::new();
        assert!(optimizer.performance_metrics.is_empty());
        assert!(optimizer.best_paths.is_empty());
        assert!(optimizer.trade_history_for_rl.is_empty());
        assert!(optimizer.q_table.is_empty());
        assert_eq!(optimizer.learning_rate_rl, 0.1);
        assert_eq!(optimizer.discount_factor_rl, 0.9);
        assert_eq!(optimizer.exploration_rate_rl, 0.1);
    }

    #[test]
    fn test_rl_state_hashing_and_equality() {
        let state1 = RLState { market_trend: 1, market_volatility: 0, recent_path_performance: 1 };
        let state2 = RLState { market_trend: 1, market_volatility: 0, recent_path_performance: 1 };
        let state3 = RLState { market_trend: -1, market_volatility: 0, recent_path_performance: 1 };

        let mut map = HashMap::new();
        map.insert(state1, "value1");

        assert_eq!(state1, state2);
        assert_ne!(state1, state3);
        assert!(map.contains_key(&state2));
        assert!(!map.contains_key(&state3));
    }

    #[test]
    fn test_discretization_functions() {
        assert_eq!(discretize_volatility(0.01), 0); // Low
        assert_eq!(discretize_volatility(0.03), 1); // Medium
        assert_eq!(discretize_volatility(0.06), 2); // High

        assert_eq!(discretize_trend(-0.5), -1); // Down
        assert_eq!(discretize_trend(0.0), 0);   // Neutral
        assert_eq!(discretize_trend(0.5), 1);   // Up
    }

    #[test]
    fn test_update_q_table_basic() {
        let mut optimizer = StrategyOptimizer::new();
        let state = RLState { market_trend: 0, market_volatility: 1, recent_path_performance: 0 };
        let action = RLAction::ExecutePath("path_test".to_string());
        let reward = 10.0;
        let next_state = state; // Simplified: next state Q value is 0

        optimizer.update_q_table(&state, &action, reward, &next_state);

        let expected_q_value = 0.0 + optimizer.learning_rate_rl * (reward + optimizer.discount_factor_rl * 0.0 - 0.0);
        // Q(s,a) = 0 + 0.1 * (10.0 + 0.9 * 0 - 0) = 0.1 * 10.0 = 1.0

        let q_value = optimizer.q_table.get(&state).unwrap().get(&action).unwrap();
        assert!((q_value - expected_q_value).abs() < 1e-6, "Q-value not updated as expected.");

        // Second update
        let reward2 = 20.0;
        optimizer.update_q_table(&state, &action, reward2, &next_state);
        // Q(s,a) = 1.0 + 0.1 * (20.0 + 0.9 * 0 - 1.0) = 1.0 + 0.1 * 19.0 = 1.0 + 1.9 = 2.9
        let expected_q_value2 = expected_q_value + optimizer.learning_rate_rl * (reward2 - expected_q_value);
        let q_value2 = optimizer.q_table.get(&state).unwrap().get(&action).unwrap();
        assert!((q_value2 - expected_q_value2).abs() < 1e-6, "Q-value not updated correctly on second call.");
    }

    #[test]
    fn test_choose_action_logic() {
        let mut optimizer = StrategyOptimizer::new();
        let state = RLState { market_trend: 0, market_volatility: 0, recent_path_performance: 0 };
        let path1_id = "path1".to_string();
        let path2_id = "path2".to_string();
        let action1_exec = RLAction::ExecutePath(path1_id.clone());
        let action2_exec = RLAction::ExecutePath(path2_id.clone());

        let mut actions_for_state = HashMap::new();
        actions_for_state.insert(action1_exec.clone(), 10.0); // path1 is good
        actions_for_state.insert(action2_exec.clone(), 5.0);  // path2 is okay
        optimizer.q_table.insert(state, actions_for_state);

        let available_paths = vec![path1_id.clone(), path2_id.clone()];

        // Test exploitation
        optimizer.exploration_rate_rl = 0.0; // Force exploitation
        let chosen_action_exploit = optimizer.choose_action(&state, &available_paths);
        assert_eq!(chosen_action_exploit, action1_exec, "Exploitation should choose action with highest Q-value.");

        // Test exploration (hard to test deterministically, but can check if it deviates)
        optimizer.exploration_rate_rl = 1.0; // Force exploration
        let mut exploration_chose_something_else = false;
        for _ in 0..20 { // Run a few times
            let chosen_action_explore = optimizer.choose_action(&state, &available_paths);
            // It should pick one of the available paths to Execute or Avoid
            match chosen_action_explore {
                RLAction::ExecutePath(pid) => assert!(available_paths.contains(&pid)),
                RLAction::AvoidPath(pid) => assert!(available_paths.contains(&pid)),
                RLAction::Hold => panic!("Exploration chose Hold, which is not an active exploration choice here."),
            }
            if chosen_action_explore != action1_exec { // If it ever picks something other than the 'best' known one
                exploration_chose_something_else = true;
            }
        }
        // With 100% exploration over many trials, it's very likely to have chosen something else at least once.
        // This is a probabilistic test, might occasionally fail but indicates exploration if usually passes.
        // For this setup, it will pick randomly between ExecutePath(path1), AvoidPath(path1), ExecutePath(path2), AvoidPath(path2)
        // So, it will often choose something other than action1_exec.
         assert!(exploration_chose_something_else, "Exploration should allow choosing non-optimal actions.");
    }


    #[test]
    fn test_add_or_update_path() {
        let mut optimizer = StrategyOptimizer::new();
        let path1 = sample_path1();
        optimizer.add_or_update_path(path1.clone());
        assert_eq!(optimizer.best_paths.len(), 1);
        assert_eq!(optimizer.best_paths.get("PATH001").unwrap().id, path1.id);

        let updated_path1 = TradingPath { expected_profit: 0.06, ..path1 };
        optimizer.add_or_update_path(updated_path1.clone());
        assert_eq!(optimizer.best_paths.len(), 1);
        assert_eq!(optimizer.best_paths.get("PATH001").unwrap().expected_profit, 0.06);
    }

    #[test]
    fn test_track_trade_and_metrics_update() {
        let mut optimizer = StrategyOptimizer::new();
        let path1 = sample_path1();
        optimizer.add_or_update_path(path1);

        let outcome1 = TradeOutcome::new("PATH001".to_string(), 0.10, sample_market_snapshot(), 1600000000); // Profit
        optimizer.track_trade(outcome1);

        assert_eq!(optimizer.trade_history_for_rl.len(), 1);
        let path_stats = optimizer.best_paths.get("PATH001").unwrap();
        assert_eq!(path_stats.times_traded, 1);
        assert_eq!(path_stats.total_profit_generated, 0.10);
        assert_eq!(path_stats.success_rate, 1.0); // (0.7 * 0 + 1) / 1 = 1.0

        assert_eq!(*optimizer.performance_metrics.get("total_trades").unwrap(), 1.0);
        assert_eq!(*optimizer.performance_metrics.get("total_profit").unwrap(), 0.10);
        assert_eq!(*optimizer.performance_metrics.get("average_profit_per_trade").unwrap(), 0.10);
        assert_eq!(*optimizer.performance_metrics.get("win_rate").unwrap(), 1.0);

        let outcome2 = TradeOutcome::new("PATH001".to_string(), -0.05, sample_market_snapshot(), 1600000010); // Loss
        optimizer.track_trade(outcome2);

        assert_eq!(optimizer.trade_history_for_rl.len(), 2);
        let path_stats2 = optimizer.best_paths.get("PATH001").unwrap();
        assert_eq!(path_stats2.times_traded, 2);
        assert_eq!(path_stats2.total_profit_generated, 0.05); // 0.10 - 0.05
        assert_eq!(path_stats2.success_rate, 0.5); // (1.0 * 1 + 0) / 2 = 0.5

        assert_eq!(*optimizer.performance_metrics.get("total_trades").unwrap(), 2.0);
        assert_eq!(*optimizer.performance_metrics.get("total_profit").unwrap(), 0.05);
        assert_eq!(*optimizer.performance_metrics.get("average_profit_per_trade").unwrap(), 0.025); // 0.05 / 2
        assert_eq!(*optimizer.performance_metrics.get("win_rate").unwrap(), 0.5); // 1 win / 2 trades
    }

    #[test]
    fn test_track_trade_unknown_path() {
        let mut optimizer = StrategyOptimizer::new();
        // PATH_UNKNOWN is not added to best_paths
        let outcome = TradeOutcome::new("PATH_UNKNOWN".to_string(), 0.10, sample_market_snapshot(), 1600000000);
        optimizer.track_trade(outcome);

        // Path stats should not be updated as it doesn't exist in the map
        assert!(optimizer.best_paths.get("PATH_UNKNOWN").is_none());
        // History and metrics should still be updated
        assert_eq!(optimizer.trade_history_for_rl.len(), 1);
        assert_eq!(*optimizer.performance_metrics.get("total_trades").unwrap(), 1.0);
    }


    #[test]
    fn test_get_best_paths() {
        let mut optimizer = StrategyOptimizer::new();
        let mut path1 = sample_path1(); // avg_profit initially 0
        let mut path2 = sample_path2(); // avg_profit initially 0

        optimizer.add_or_update_path(path1.clone());
        optimizer.add_or_update_path(path2.clone());

        // Trade for path1 to make it profitable
        let outcome_p1_good = TradeOutcome::new("PATH001".to_string(), 0.20, sample_market_snapshot(), 1);
        optimizer.track_trade(outcome_p1_good); // path1 avg profit: 0.20

        // Trades for path2
        let outcome_p2_good = TradeOutcome::new("PATH002".to_string(), 0.50, sample_market_snapshot(), 2); // High profit
        optimizer.track_trade(outcome_p2_good);
        let outcome_p2_bad = TradeOutcome::new("PATH002".to_string(), -0.10, sample_market_snapshot(), 3); // Small loss
        optimizer.track_trade(outcome_p2_bad); // path2 avg profit: (0.50 - 0.10) / 2 = 0.20

        // At this point, path1 avg profit is 0.20 (1 trade), path2 avg profit is 0.20 (2 trades)
        // The sort is stable, so original insertion order might affect if averages are identical.
        // Let's make path1 slightly better
        path1 = optimizer.best_paths.get("PATH001").unwrap().clone();
        path1.total_profit_generated = 0.21; // Manually adjust for test clarity, avg = 0.21
        optimizer.add_or_update_path(path1);


        let best_1 = optimizer.get_best_paths(1);
        assert_eq!(best_1.len(), 1);
        assert_eq!(best_1[0].id, "PATH001"); // PATH001 has avg profit 0.21

        let best_2 = optimizer.get_best_paths(2);
        assert_eq!(best_2.len(), 2);
        assert_eq!(best_2[0].id, "PATH001"); // PATH001 avg profit 0.21
        assert_eq!(best_2[1].id, "PATH002"); // PATH002 avg profit 0.20

        // Test with N > available paths
        let best_3 = optimizer.get_best_paths(3);
        assert_eq!(best_3.len(), 2);
    }

    #[test]
    fn test_trading_path_average_profit() {
        let mut path = TradingPath::new("TEST".to_string(), vec![], 0.0, 0.0);
        assert_eq!(path.average_profit(), 0.0);
        path.total_profit_generated = 10.0;
        path.times_traded = 2;
        assert_eq!(path.average_profit(), 5.0);
    }

    #[test]
    fn test_optimize_strategies_placeholder() { // Renamed to reflect new signature
        let mut optimizer = StrategyOptimizer::new();
        optimizer.optimize_strategies(None); // Call with no current market snapshot

        let current_market = MarketStateSnapshot::new(0.3, 0.8);
        optimizer.optimize_strategies(Some(current_market));
    }

    #[test]
    fn test_adapt_strategy_based_on_market_placeholder() {
        let mut optimizer = StrategyOptimizer::new();
        let market_snapshot = sample_market_snapshot();
        optimizer.adapt_strategy_based_on_market(&market_snapshot);

        let high_vol_snapshot = MarketStateSnapshot::new(0.8, 0.0);
        optimizer.adapt_strategy_based_on_market(&high_vol_snapshot);
    }

    #[test]
    fn test_optimize_strategies_rl_logging() {
        let mut optimizer = StrategyOptimizer::new();
        let path1 = sample_path1();
        optimizer.add_or_update_path(path1);

        // Simulate a trade to have some history for market snapshot
        let outcome1 = TradeOutcome::new("PATH001".to_string(), 0.10, sample_market_snapshot(), 1600000000);
        optimizer.track_trade(outcome1);

        // Call with some current market state
        let current_market = MarketStateSnapshot::new(0.3, 0.8); // Example current state
        optimizer.optimize_strategies(Some(current_market));
        // Check logs manually for "RL Preference for path PATH001..."

        // Call without current market state (should use last from history)
        optimizer.optimize_strategies(None);
        // Check logs manually
    }
}
