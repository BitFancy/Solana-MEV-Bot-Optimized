# Solana MEV Bot

Welcome to the **Solana Arbitrage Bot**! This Rust-based bot is designed for executing Maximal Extractable Value (MEV) strategies on the Solana blockchain.

Ensure you have the following installed:

- **Rust**: Install Rust using rustup:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
  
- **Solana CLI**: Install by running:
  ```bash
  sh -c "$(curl -sSfL https://release.solana.com/stable/install)"
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BitFancy/solana-mev-bot-optimized-mine.git
   cd solana-mev-bot-optimized-mine
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. Configure your environment:
   - Create a `.env` file in the root directory and add your Solana wallet private key and RPC link:
     ```
     SOLANA_WALLET_PRIVATE_KEY=YOUR_PRIVATE_KEY
     DEFAULT_RPC=https://api.mainnet-beta.solana.com
     ```

## Running the Project

To run the bot, use the following command:

```bash
cargo run --release
```

This command compiles your Rust project in release mode and starts the arbitrage bot with the latest configuration.
