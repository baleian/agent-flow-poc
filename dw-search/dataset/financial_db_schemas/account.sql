CREATE TABLE IF NOT EXISTS account (
    account_id INT COMMENT 'The ID of the account',
    district_id INT COMMENT 'Location of the branch',
    frequency STRING COMMENT 'Frequency of the account. "POPLATEK MESICNE" stands for monthly issuance, "POPLATEK TYDNE" for weekly issuance, and "POPLATEK PO OBRATU" for issuance after transaction.',
    date STRING COMMENT 'The creation date of the account, in the form YYMMDD.'
)
COMMENT 'Table containing customer account information.'
