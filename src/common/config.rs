use std::env;
use crate::arbitrage::constants::{Env, get_env};

#[derive(Debug)]
pub struct Config {
    pub env: Env,
}

impl Config {
    /// Creates a new Config instance by loading environment variables.
    pub fn new() -> Self {
        let env = Env::new();
        Config { env }
    }

    /// Loads a specific configuration value from the environment.
    pub fn load_value(&self, key: &str) -> String {
        get_env(key)
    }

    /// Displays the current configuration for debugging purposes.
    pub fn display(&self) {
        println!("Current Configuration:");
        println!("Block Engine URL: {}", self.env.block_engine_url);
        println!("Mainnet RPC URL: {}", self.env.mainnet_rpc_url);
        println!("Devnet RPC URL: {}", self.env.devnet_rpc_url);
        // Add more fields as necessary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_loading() {
        // Set up test environment variables
        env::set_var("BLOCK_ENGINE_URL", "http://localhost:8080");
        env::set_var("MAINNET_RPC_URL", "https://api.mainnet-beta.solana.com");
        
        let config = Config::new();
        
        assert_eq!(config.env.block_engine_url, "http://localhost:8080");
        assert_eq!(config.env.mainnet_rpc_url, "https://api.mainnet-beta.solana.com");
    }
}