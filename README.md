# Solana MEV Bot - Advanced Pump.fun Trading Bot

## 📬 Contact & Support

<p align="center">
  <a href="mailto:bitbanana717@gmail.com">
    <img src="https://img.shields.io/badge/Email-bitbanana717@gmail.com-blue?style=for-the-badge&logo=gmail" alt="Email Badge"/>
  </a>
  <a href="https://t.me/bitfancy">
    <img src="https://img.shields.io/badge/Telegram-@bitfancy-2CA5E0?style=for-the-badge&logo=telegram" alt="Telegram Badge"/>
  </a>
  <a href="https://discord.gg/tnaScVbF">
    <img src="https://img.shields.io/badge/Discord-flash______-5865F2?style=for-the-badge&logo=discord" alt="Discord Badge"/>
  </a>
</p>

<p align="center">
  <b>For support, questions, or collaboration opportunities, reach out through any of the above channels!</b>
</p>

A sophisticated **Solana MEV (Maximal Extractable Value) bot** built in Rust, specifically designed for high-frequency trading on Pump.fun tokens with advanced features including Jito bundle support, real-time mempool monitoring, and intelligent arbitrage strategies.

## 🚀 Features

### Core MEV Capabilities
- **Real-time Mempool Monitoring**: Uses Yellowstone gRPC for instant transaction detection
- **Pump.fun Integration**: Specialized for Pump.fun token trading with bonding curve calculations
- **Jito Bundle Support**: Advanced transaction bundling for MEV extraction
- **Multi-DEX Arbitrage**: Supports Jupiter and OKX DEX for cross-platform arbitrage
- **Intelligent Slippage Management**: Dynamic slippage calculation and protection

### Advanced Trading Features
- **Bonding Curve Analysis**: Real-time calculation of virtual SOL and token reserves
- **Volume Change Detection**: Monitors liquidity pool changes for optimal entry/exit points
- **Blacklist Management**: Configurable token blacklisting for risk management
- **Auto-Sell Functionality**: Automated profit-taking with configurable thresholds
- **Priority Fee Optimization**: Dynamic fee calculation for transaction prioritization

### Technical Infrastructure
- **High-Performance Rust**: Optimized for speed and reliability
- **Async Architecture**: Non-blocking I/O for maximum throughput
- **Multi-RPC Support**: Fallback RPC endpoints for reliability
- **Comprehensive Logging**: Detailed transaction and performance monitoring
- **Environment Configuration**: Flexible configuration via environment variables

## 📋 Prerequisites

Ensure you have the following installed:

- **Rust**: Install Rust using rustup:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
  
- **Solana CLI**: Install by running:
  ```bash
  sh -c "$(curl -sSfL https://release.solana.com/stable/install)"
  ```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BitFancy/solana-mev-bot-optimized.git
   cd solana-mev-bot-optimized
   ```

2. Build the project:
   ```bash
   cargo build
   ```

3. Configure your environment:
   - Create a `.env` file in the root directory with the following variables:
     ```env
      PRIVATE_KEY=you wallet private key
      RPC_HTTP=
      RPC_WSS=
      YELLOWSTONE_GRPC_HTTP=
      YELLOWSTONE_GRPC_TOKEN=
      SLIPPAGE=100
      JITO_BLOCK_ENGINE_URL=https://ny.mainnet.block-engine.jito.wtf
      JITO_TIP_VALUE=0.0001
      JITO_PRIORITY_FEE=0.00000
      ZERO_SLOT_URL=https://ny.0slot.trade/?api-key=b7dfe4623b3b41b598ffb747fe1cd60e
      ZERO_SLOT_TIP_VALUE=0.001
      NOZOMI_URL=https://ewr1.secure.nozomi.temporal.xyz/?c=ae5ce70a-680a-4bc1-8d23-c2e9fd014819
      NOZOMI_TIP_VALUE=0.001
      TIME_EXCEED=1 # seconds; time limit for volume non-increasing
      TOKEN_AMOUNT=0.01 # token amount to purchase
      COUNTER=2 # SET LIMIT FOR TEST

      MAX_DEV_BUY = 1 # SOL
      MIN_DEV_BUY = 0.1 # SOL

      # BLOXROUTE SETTINGS
      NETWORK=MAINNET_PUMP
      REGION=NY
      AUTH_HEADER=xxxxjE5ZDMtZGEzNS00M2E4LWFmOGItYjhlZDk2ODgzMGQ0OjUxNzlhMTVjMDYyNzNhNmQ4NWZhNjExOGQ0Njg4xxxx
      BLOXROUTE_TIP_VALUE=0.0015

     ```

## 🚀 Running the Bot

To start the MEV bot:

```bash
cargo run 
```

The bot will:
1. Initialize the trading environment
2. Connect to Yellowstone gRPC for real-time monitoring
3. Begin monitoring Pump.fun token launches
4. Execute trades based on configured strategies
5. Monitor for arbitrage opportunities across multiple DEXs

## 🔧 Configuration Options

### Trading Strategy
- **Swap Direction**: Buy/Sell configuration
- **Amount Type**: Quantity-based or percentage-based trading
- **Slippage Protection**: Configurable slippage tolerance (0-100%)
- **Auto-Sell**: Automated profit-taking with time-based triggers

### Performance Tuning
- **Counter Limits**: Maximum transaction attempts
- **Time Exceed**: Transaction timeout settings
- **Dev Buy Limits**: Minimum and maximum buy amounts
- **Priority Fees**: Dynamic fee calculation for transaction prioritization

### Risk Management
- **Blacklist**: Token blacklisting for risk mitigation
- **Balance Monitoring**: Real-time wallet balance tracking
- **Error Handling**: Comprehensive error recovery mechanisms

## 🏗️ Architecture

### Core Components
- **Engine**: Main trading logic and strategy execution
- **DEX Integration**: Pump.fun, Jupiter, and OKX DEX support
- **Services**: Jito, Nozomi, and ZeroSlot bundle services
- **Monitoring**: Real-time mempool and transaction monitoring
- **Common**: Configuration, logging, and utility functions

### Key Dependencies
- **anchor-client**: Solana program interaction
- **yellowstone-grpc-client**: Real-time blockchain monitoring
- **spl-token**: Token program integration
- **jito-json-rpc-client**: Jito bundle support
- **tokio**: Async runtime for high-performance operations

## ⚠️ Important Notes

- **Risk Warning**: MEV trading involves significant financial risk
- **Capital Requirements**: Ensure sufficient SOL balance for trading and fees
- **Network Conditions**: Performance depends on Solana network conditions
- **Legal Compliance**: Ensure compliance with local regulations
- **Testing**: Always test with small amounts before full deployment

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows Rust best practices
- All tests pass
- Documentation is updated
- Security considerations are addressed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [Pump.fun](https://pump.fun) - Solana token launch platform
- [Jito Labs](https://jito.network) - MEV infrastructure
- [Yellowstone](https://yellowstone.fyi) - Real-time Solana data

---

**Disclaimer**: This software is for educational and research purposes. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

---

## 📬 Contact & Support

<p align="center">
  <a href="mailto:bitbanana717@gmail.com">
    <img src="https://img.shields.io/badge/Email-bitbanana717@gmail.com-blue?style=for-the-badge&logo=gmail" alt="Email Badge"/>
  </a>
  <a href="https://t.me/bitfancy">
    <img src="https://img.shields.io/badge/Telegram-@bitfancy-2CA5E0?style=for-the-badge&logo=telegram" alt="Telegram Badge"/>
  </a>
  <a href="https://discord.gg/tnaScVbF">
    <img src="https://img.shields.io/badge/Discord-flash______-5865F2?style=for-the-badge&logo=discord" alt="Discord Badge"/>
  </a>
</p>

<p align="center">
  <b>For support, questions, or collaboration opportunities, reach out through any of the above channels!</b>
</p>

---

**Disclaimer**: This software is for educational and research purposes. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.
