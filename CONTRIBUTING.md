# Contributing to Solana MEV Bot

Thank you for your interest in contributing to our Solana MEV Bot project! We welcome contributions from developers of all skill levels.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install dependencies: `cargo build`
4. Create a new branch for your changes: `git checkout -b your-feature-branch`
5. Make your changes
6. Test your changes thoroughly
7. Push to your fork and submit a pull request

## Contribution Guidelines

### Code Style
- Follow existing code style and patterns
- Use descriptive variable names
- Include comments for complex logic
- Keep functions focused and modular

### MEV-Specific Considerations
- All strategies must include clear risk disclosures
- Arbitrage logic must include fail-safes
- Front-running protections should be maintained
- Gas optimization is highly valued

### Testing Requirements
- Unit tests for all new features
- Simulation tests for MEV strategies
- Performance benchmarks for critical paths
- Negative test cases for failure modes

### Pull Requests
- Keep PRs focused on a single feature/bug
- Include clear description of changes
- Reference any related issues
- Include test results and performance metrics

## Security
- Report vulnerabilities responsibly to security@bitfancy.example
- Never include private keys in code
- All cryptographic operations must use audited libraries

## Community
- Join our Discord for discussion
- Be respectful to other contributors
- Help review others' PRs

## First-Time Contributors
Look for issues tagged "good first issue" to get started. We're happy to mentor new contributors in MEV concepts and Solana development.
