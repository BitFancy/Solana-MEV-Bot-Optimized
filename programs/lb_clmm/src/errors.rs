use anchor_lang::prelude::*;

#[error_code]
#[derive(PartialEq)]
pub enum LBError {
    #[msg("Invalid start bin index")]
    InvalidStartBinIndex,

    #[msg("Invalid bin id")]
    InvalidBinId,

    #[msg("Invalid input data")]
    InvalidInput,

    #[msg("Exceeded amount slippage tolerance")]
    ExceededAmountSlippageTolerance,

    #[msg("Exceeded bin slippage tolerance")]
    ExceededBinSlippageTolerance,

    #[msg("Composition factor flawed")]
    CompositionFactorFlawed,

    #[msg("Non preset bin step")]
    NonPresetBinStep,

    #[msg("Zero liquidity")]
    ZeroLiquidity,

    #[msg("Invalid position")]
    InvalidPosition,

    #[msg("Bin array not found")]
    BinArrayNotFound,

    #[msg("Invalid token mint")]
    InvalidTokenMint,

    #[msg("Invalid account for single deposit")]
    InvalidAccountForSingleDeposit,

    #[msg("Pair insufficient liquidity")]
    PairInsufficientLiquidity,

    #[msg("Invalid fee owner")]
    InvalidFeeOwner,

    #[msg("Invalid fee withdraw amount")]
    InvalidFeeWithdrawAmount,

    #[msg("Invalid admin")]
    InvalidAdmin,

    #[msg("Identical fee owner")]
    IdenticalFeeOwner,

    #[msg("Invalid basis point")]
    InvalidBps,

    #[msg("Math operation overflow")]
    MathOverflow,

    #[msg("Type cast error")]
    TypeCastFailed,

    #[msg("Invalid reward index")]
    InvalidRewardIndex,

    #[msg("Invalid reward duration")]
    InvalidRewardDuration,

    #[msg("Reward already initialized")]
    RewardInitialized,

    #[msg("Reward not initialized")]
    RewardUninitialized,

    #[msg("Identical funder")]
    IdenticalFunder,

    #[msg("Reward campaign in progress")]
    RewardCampaignInProgress,

    #[msg("Reward duration is the same")]
    IdenticalRewardDuration,

    #[msg("Invalid bin array")]
    InvalidBinArray,

    #[msg("Bin arrays must be continuous")]
    NonContinuousBinArrays,

    #[msg("Invalid reward vault")]
    InvalidRewardVault,

    #[msg("Position is not empty")]
    NonEmptyPosition,

    #[msg("Unauthorized access")]
    UnauthorizedAccess,

    #[msg("Invalid fee parameter")]
    InvalidFeeParameter,

    #[msg("Missing oracle account")]
    MissingOracle,

    #[msg("Insufficient observation sample")]
    InsufficientSample,

    #[msg("Invalid lookup timestamp")]
    InvalidLookupTimestamp,

    #[msg("Bitmap extension account is not provided")]
    BitmapExtensionAccountIsNotProvided,

    #[msg("Cannot find non-zero liquidity binArrayId")]
    CannotFindNonZeroLiquidityBinArrayId,

    #[msg("Bin id out of bound")]
    BinIdOutOfBound,

    #[msg("Insufficient amount in for minimum out")]
    InsufficientOutAmount,

    #[msg("Invalid position width")]
    InvalidPositionWidth,

    #[msg("Excessive fee update")]
    ExcessiveFeeUpdate,

    #[msg("Pool disabled")]
    PoolDisabled,

    #[msg("Invalid pool type")]
    InvalidPoolType,

    #[msg("Whitelist for wallet is full")]
    ExceedMaxWhitelist,

    #[msg("Invalid index")]
    InvalidIndex,

    #[msg("Reward not ended")]
    RewardNotEnded,

    #[msg("Must withdraw ineligible reward")]
    MustWithdrawnIneligibleReward,

    #[msg("Invalid strategy parameters")]
    InvalidStrategyParameters,

    #[msg("Liquidity locked")]
    LiquidityLocked,

    #[msg("Invalid lock release slot")]
    InvalidLockReleaseSlot,
}
