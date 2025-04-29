use crate::assert_eq_admin;
use crate::errors::LBError;
use crate::state::preset_parameters::PresetParameter;
use crate::utils::seeds::PRESET_PARAMETER;
use anchor_lang::prelude::*;

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct InitPresetParametersIx {
    /// Bin step. Represent the price increment / decrement.
    pub bin_step: u16,
    /// Used for base fee calculation. base_fee_rate = base_factor * bin_step
    pub base_factor: u16,
    /// Filter period determine high frequency trading time window.
    pub filter_period: u16,
    /// Decay period determine when the volatile fee start decay / decrease.
    pub decay_period: u16,
    /// Reduction factor controls the volatile fee rate decrement rate.
    pub reduction_factor: u16,
    /// Used to scale the variable fee component depending on the dynamic of the market
    pub variable_fee_control: u32,
    /// Maximum number of bin crossed can be accumulated. Used to cap volatile fee rate.
    pub max_volatility_accumulator: u32,
    /// Min bin id supported by the pool based on the configured bin step.
    pub min_bin_id: i32,
    /// Max bin id supported by the pool based on the configured bin step.
    pub max_bin_id: i32,
    /// Portion of swap fees retained by the protocol by controlling protocol_share parameter. protocol_swap_fee = protocol_share * total_swap_fee
    pub protocol_share: u16,
}

#[derive(Accounts)]
#[instruction(ix: InitPresetParametersIx)]
pub struct InitializePresetParameter<'info> {
    #[account(
        init,
        seeds = [
            PRESET_PARAMETER,
            &ix.bin_step.to_le_bytes(),
            &ix.base_factor.to_le_bytes()
        ],
        bump,
        payer = admin,
        space = 8 + PresetParameter::INIT_SPACE
    )]
    pub preset_parameter: Account<'info, PresetParameter>,

    #[account(
        mut,
        constraint = assert_eq_admin(admin.key()) @ LBError::InvalidAdmin
    )]
    pub admin: Signer<'info>,

    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

pub fn handle(ctx: Context<InitializePresetParameter>, ix: InitPresetParametersIx) -> Result<()> {
    Ok(())
}
