#![allow(warnings)]

use anchor_lang::prelude::*;

pub mod constants;
pub mod errors;
pub mod events;
pub mod instructions;
pub mod manager;
pub mod math;
pub mod state;
pub mod utils;

use instructions::add_liquidity::*;
use instructions::add_liquidity_by_strategy::*;
use instructions::add_liquidity_by_strategy_one_side::*;
use instructions::add_liquidity_by_weight::*;
use instructions::add_liquidity_by_weight_one_side::*;
use instructions::add_liquidity_single_side_precise::*;
use instructions::claim_fee::*;
use instructions::claim_reward::*;
use instructions::close_position::*;
use instructions::close_preset_parameter::*;
use instructions::fund_reward::*;
use instructions::increase_oracle_length::*;
use instructions::initialize_bin_array::*;
use instructions::initialize_bin_array_bitmap_extension::*;
use instructions::initialize_lb_pair::*;
use instructions::initialize_permission_lb_pair::*;
use instructions::initialize_position::*;
use instructions::initialize_position_by_operator::*;
use instructions::initialize_position_pda::*;
use instructions::initialize_preset_parameters::*;
use instructions::initialize_reward::*;
use instructions::migrate_bin_array::*;
use instructions::migrate_position::*;
use instructions::position_authorize::*;
use instructions::remove_liquidity::*;
use instructions::set_activation_slot::*;
use instructions::set_lock_release_slot::*;
use instructions::set_pre_activation_slot_duration::*;
use instructions::set_pre_activation_swap_address::*;
use instructions::swap::*;
use instructions::toggle_pair_status::*;
use instructions::update_fee_owner::*;
use instructions::update_fee_parameters::*;
use instructions::update_fees_and_rewards::*;
use instructions::update_position_operator::*;
use instructions::update_reward_duration::*;
use instructions::update_reward_funder::*;
use instructions::update_whitelisted_wallet::*;
use instructions::withdraw_ineligible_reward::*;
use instructions::withdraw_protocol_fee::*;

#[cfg(feature = "localnet")]
declare_id!("LbVRzDTvBDEcrthxfZ4RL6yiq3uZw8bS6MwtdY6UhFQ");

#[cfg(feature = "staging")]
declare_id!("tLBro6JJuZNnpoad3p8pXKohE9f7f7tBZJpaeh6pXt1");

#[cfg(not(any(feature = "localnet", feature = "staging")))]
declare_id!("LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo");

pub mod admin {
    use super::*;
    use anchor_lang::solana_program::pubkey;

    #[cfg(feature = "localnet")]
    pub const ADMINS: [Pubkey; 1] = [pubkey!("bossj3JvwiNK7pvjr149DqdtJxf2gdygbcmEPTkb2F1")];

    #[cfg(not(feature = "localnet"))]
    pub const ADMINS: [Pubkey; 3] = [
        pubkey!("5unTfT2kssBuNvHPY6LbJfJpLqEcdMxGYLWHwShaeTLi"),
        pubkey!("ChSAh3XXTxpp5n2EmgSCm6vVvVPoD1L9VrK3mcQkYz7m"),
        pubkey!("DHLXnJdACTY83yKwnUkeoDjqi4QBbsYGa1v8tJL76ViX"),
    ];
}

pub mod launch_pool_config_admins {
    use super::*;
    use anchor_lang::solana_program::pubkey;

    #[cfg(feature = "localnet")]
    pub const ADMINS: [Pubkey; 1] = [pubkey!("bossj3JvwiNK7pvjr149DqdtJxf2gdygbcmEPTkb2F1")];

    #[cfg(not(feature = "localnet"))]
    pub const ADMINS: [Pubkey; 4] = [
        pubkey!("4Qo6nr3CqiynvnA3SsbBtzVT3B1pmqQW4dwf2nFmnzYp"),
        pubkey!("5unTfT2kssBuNvHPY6LbJfJpLqEcdMxGYLWHwShaeTLi"),
        pubkey!("ChSAh3XXTxpp5n2EmgSCm6vVvVPoD1L9VrK3mcQkYz7m"),
        pubkey!("DHLXnJdACTY83yKwnUkeoDjqi4QBbsYGa1v8tJL76ViX"),
    ];
}

/// Authorized pubkey to withdraw protocol fee
pub mod fee_owner {
    use super::*;

    #[cfg(feature = "localnet")]
    declare_id!("bossj3JvwiNK7pvjr149DqdtJxf2gdygbcmEPTkb2F1");

    #[cfg(not(feature = "localnet"))]
    declare_id!("6WaLrrRfReGKBYUSkmx2K6AuT21ida4j8at2SUiZdXu8");
}

pub fn assert_eq_admin(admin: Pubkey) -> bool {
    crate::admin::ADMINS
        .iter()
        .any(|predefined_admin| predefined_admin.eq(&admin))
}

pub fn assert_eq_launch_pool_admin(admin: Pubkey) -> bool {
    crate::launch_pool_config_admins::ADMINS
        .iter()
        .any(|predefined_launch_pool_admin| predefined_launch_pool_admin.eq(&admin))
}

#[program]
pub mod lb_clmm {

    use self::instructions::add_liquidity_single_side_precise::CompressedBinDepositAmount;

    use super::*;

    pub fn initialize_lb_pair(
        ctx: Context<InitializeLbPair>,
        active_id: i32,
        bin_step: u16,
    ) -> Result<()> {
        instructions::initialize_lb_pair::handle(ctx, active_id, bin_step)
    }

    pub fn initialize_permission_lb_pair(
        ctx: Context<InitializePermissionLbPair>,
        ix_data: InitPermissionPairIx,
    ) -> Result<()> {
        instructions::initialize_permission_lb_pair::handle(ctx, ix_data)
    }

    pub fn initialize_bin_array_bitmap_extension(
        ctx: Context<InitializeBinArrayBitmapExtension>,
    ) -> Result<()> {
        instructions::initialize_bin_array_bitmap_extension::handle(ctx)
    }

    pub fn initialize_bin_array(ctx: Context<InitializeBinArray>, index: i64) -> Result<()> {
        instructions::initialize_bin_array::handle(ctx, index)
    }

    pub fn add_liquidity<'a, 'b, 'c, 'info>(
        ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidity<'info>>,
        liquidity_parameter: LiquidityParameter,
    ) -> Result<()> {
        instructions::add_liquidity::handle(ctx, liquidity_parameter)
    }
    pub fn add_liquidity_by_weight<'a, 'b, 'c, 'info>(
        ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidity<'info>>,
        liquidity_parameter: LiquidityParameterByWeight,
    ) -> Result<()> {
        instructions::add_liquidity_by_weight::handle(&ctx, &liquidity_parameter)
    }

    pub fn add_liquidity_by_strategy<'a, 'b, 'c, 'info>(
        ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidity<'info>>,
        liquidity_parameter: LiquidityParameterByStrategy,
    ) -> Result<()> {
        instructions::add_liquidity_by_strategy::handle(ctx, &liquidity_parameter)
    }

    pub fn add_liquidity_by_strategy_one_side<'a, 'b, 'c, 'info>(
        ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidityOneSide<'info>>,
        liquidity_parameter: LiquidityParameterByStrategyOneSide,
    ) -> Result<()> {
        instructions::add_liquidity_by_strategy_one_side::handle(ctx, &liquidity_parameter)
    }

    pub fn add_liquidity_one_side<'a, 'b, 'c, 'info>(
        ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidityOneSide<'info>>,
        liquidity_parameter: LiquidityOneSideParameter,
    ) -> Result<()> {
        instructions::add_liquidity_by_weight_one_side::handle(&ctx, &liquidity_parameter)
    }

    pub fn remove_liquidity<'a, 'b, 'c, 'info>(
        ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidity<'info>>,
        bin_liquidity_removal: Vec<BinLiquidityReduction>,
    ) -> Result<()> {
        instructions::remove_liquidity::handle(ctx, bin_liquidity_removal)
    }

    pub fn initialize_position(
        ctx: Context<InitializePosition>,
        lower_bin_id: i32,
        width: i32,
    ) -> Result<()> {
        instructions::initialize_position::handle(ctx, lower_bin_id, width)
    }

    pub fn initialize_position_pda(
        ctx: Context<InitializePositionPda>,
        lower_bin_id: i32,
        width: i32,
    ) -> Result<()> {
        instructions::initialize_position_pda::handle(ctx, lower_bin_id, width)
    }

    pub fn initialize_position_by_operator(
        ctx: Context<InitializePositionByOperator>,
        lower_bin_id: i32,
        width: i32,
        owner: Pubkey,
        fee_owner: Pubkey,
    ) -> Result<()> {
        instructions::initialize_position_by_operator::handle(
            ctx,
            lower_bin_id,
            width,
            owner,
            fee_owner,
        )
    }

    pub fn update_position_operator(
        ctx: Context<UpdatePositionOperator>,
        operator: Pubkey,
    ) -> Result<()> {
        instructions::update_position_operator::handle(ctx, operator)
    }

    pub fn swap<'a, 'b, 'c, 'info>(
        ctx: Context<'a, 'b, 'c, 'info, Swap<'info>>,
        amount_in: u64,
        min_amount_out: u64,
    ) -> Result<()> {
        instructions::swap::handle(ctx, amount_in, min_amount_out)
    }

    pub fn withdraw_protocol_fee(
        ctx: Context<WithdrawProtocolFee>,
        amount_x: u64,
        amount_y: u64,
    ) -> Result<()> {
        instructions::withdraw_protocol_fee::handle(ctx, amount_x, amount_y)
    }

    pub fn update_fee_owner(ctx: Context<UpdateFeeOwner>) -> Result<()> {
        instructions::update_fee_owner::handle(ctx)
    }

    pub fn initialize_reward(
        ctx: Context<InitializeReward>,
        reward_index: u64,
        reward_duration: u64,
        funder: Pubkey,
    ) -> Result<()> {
        instructions::initialize_reward::handle(ctx, reward_index, reward_duration, funder)
    }

    pub fn fund_reward(
        ctx: Context<FundReward>,
        reward_index: u64,
        amount: u64,
        carry_forward: bool,
    ) -> Result<()> {
        instructions::fund_reward::handle(ctx, reward_index, amount, carry_forward)
    }

    pub fn update_reward_funder(
        ctx: Context<UpdateRewardFunder>,
        reward_index: u64,
        new_funder: Pubkey,
    ) -> Result<()> {
        instructions::update_reward_funder::handle(ctx, reward_index, new_funder)
    }

    pub fn update_reward_duration(
        ctx: Context<UpdateRewardDuration>,
        reward_index: u64,
        new_duration: u64,
    ) -> Result<()> {
        instructions::update_reward_duration::handle(ctx, reward_index, new_duration)
    }

    pub fn claim_reward(ctx: Context<ClaimReward>, reward_index: u64) -> Result<()> {
        instructions::claim_reward::handle(ctx, reward_index)
    }

    pub fn claim_fee(ctx: Context<ClaimFee>) -> Result<()> {
        instructions::claim_fee::handle(ctx)
    }

    pub fn close_position(ctx: Context<ClosePosition>) -> Result<()> {
        instructions::close_position::handle(ctx)
    }

    pub fn update_fee_parameters(
        ctx: Context<UpdateFeeParameters>,
        fee_parameter: FeeParameter,
    ) -> Result<()> {
        instructions::update_fee_parameters::handle(ctx, fee_parameter)
    }

    pub fn increase_oracle_length(
        ctx: Context<IncreaseOracleLength>,
        length_to_add: u64,
    ) -> Result<()> {
        instructions::increase_oracle_length::handle(ctx, length_to_add)
    }

    pub fn initialize_preset_parameter(
        ctx: Context<InitializePresetParameter>,
        ix: InitPresetParametersIx,
    ) -> Result<()> {
        instructions::initialize_preset_parameters::handle(ctx, ix)
    }

    pub fn close_preset_parameter(ctx: Context<ClosePresetParameter>) -> Result<()> {
        instructions::close_preset_parameter::handle(ctx)
    }

    pub fn remove_all_liquidity<'a, 'b, 'c, 'info>(
        ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidity<'info>>,
    ) -> Result<()> {
        instructions::remove_all_liquidity::handle(ctx)
    }

    pub fn toggle_pair_status(ctx: Context<TogglePairStatus>) -> Result<()> {
        instructions::toggle_pair_status::handle(ctx)
    }

    pub fn update_whitelisted_wallet(
        ctx: Context<UpdateWhitelistWallet>,
        idx: u8,
        wallet: Pubkey,
    ) -> Result<()> {
        instructions::update_whitelisted_wallet::handle(ctx, idx.into(), wallet)
    }

    pub fn migrate_position(ctx: Context<MigratePosition>) -> Result<()> {
        instructions::migrate_position::handle(ctx)
    }

    pub fn migrate_bin_array(ctx: Context<MigrateBinArray>) -> Result<()> {
        instructions::migrate_bin_array::handle(ctx)
    }

    pub fn update_fees_and_rewards(ctx: Context<UpdateFeesAndRewards>) -> Result<()> {
        instructions::update_fees_and_rewards::handle(ctx)
    }

    pub fn withdraw_ineligible_reward(
        ctx: Context<WithdrawIneligibleReward>,
        reward_index: u64,
    ) -> Result<()> {
        instructions::withdraw_ineligible_reward::handle(ctx, reward_index)
    }

    pub fn set_activation_slot(
        ctx: Context<SetActivationSlot>,
        activation_slot: u64,
    ) -> Result<()> {
        instructions::set_activation_slot::handle(ctx, activation_slot)
    }

    pub fn set_lock_release_slot(
        ctx: Context<SetLockReleaseSlot>,
        new_lock_release_slot: u64,
    ) -> Result<()> {
        instructions::set_lock_release_slot::handle(ctx, new_lock_release_slot)
    }

    pub fn add_liquidity_one_side_precise<'a, 'b, 'c, 'info>(
        ctx: Context<'a, 'b, 'c, 'info, ModifyLiquidityOneSide<'info>>,
        parameter: AddLiquiditySingleSidePreciseParameter,
    ) -> Result<()> {
        instructions::add_liquidity_single_side_precise::handle(ctx, parameter)
    }

    pub fn set_pre_activation_slot_duration(
        ctx: Context<SetPreActivationInfo>,
        pre_activation_slot_duration: u16,
    ) -> Result<()> {
        instructions::set_pre_activation_slot_duration::handle(ctx, pre_activation_slot_duration)
    }

    pub fn set_pre_activation_swap_address(
        ctx: Context<SetPreActivationInfo>,
        pre_activation_swap_address: Pubkey,
    ) -> Result<()> {
        instructions::set_pre_activation_swap_address::handle(ctx, pre_activation_swap_address)
    }
}
