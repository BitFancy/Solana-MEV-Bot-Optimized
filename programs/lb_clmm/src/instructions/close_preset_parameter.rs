use crate::assert_eq_admin;
use crate::errors::LBError;
use crate::state::preset_parameters::PresetParameter;
use anchor_lang::prelude::*;

#[derive(Accounts)]
pub struct ClosePresetParameter<'info> {
    #[account(
        mut,
        close = rent_receiver
    )]
    pub preset_parameter: Account<'info, PresetParameter>,

    #[account(
        mut,
        constraint = assert_eq_admin(admin.key()) @ LBError::InvalidAdmin
    )]
    pub admin: Signer<'info>,

    /// CHECK: Account to receive closed account rental SOL
    #[account(mut)]
    pub rent_receiver: UncheckedAccount<'info>,
}

pub fn handle(_ctx: Context<ClosePresetParameter>) -> Result<()> {
    // Anchor handle everything
    Ok(())
}
