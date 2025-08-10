CREATE TABLE IF NOT EXISTS card (
    card_id INT COMMENT 'ID number of the credit card',
    disp_id INT COMMENT 'Disposition ID',
    type STRING COMMENT 'Type of credit card. "junior": junior class, "classic": standard class, "gold": high-level credit card.',
    issued STRING COMMENT 'The date when the credit card was issued, in the form YYMMDD.'
)
COMMENT 'Table containing credit card information.'