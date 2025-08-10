CREATE TABLE IF NOT EXISTS trans (
    trans_id INT COMMENT 'Unique ID for the transaction',
    account_id INT COMMENT 'ID of the account associated with the transaction',
    date DATE COMMENT 'Date of the transaction',
    type STRING COMMENT 'Type of transaction. "PRIJEM": credit, "VYDAJ": withdrawal.',
    operation STRING COMMENT 'Mode of transaction. "VYBER KARTOU": credit card withdrawal, "VKLAD": credit in cash, "PREVOD Z UCTU": collection from another bank, "VYBER": withdrawal in cash, "PREVOD NA UCET": remittance to another bank.',
    amount INT COMMENT 'Amount of money in USD',
    balance INT COMMENT 'Balance after the transaction in USD',
    k_symbol STRING COMMENT 'Characterization of the transaction. "POJISTNE": insurance payment, "SLUZBY": payment for statement, "UROK": interest credited, "SANKC. UROK": sanction interest, "SIPO": household, "DUCHOD": old-age pension, "UVER": loan payment.',
    bank STRING COMMENT 'Bank of the partner (two-letter code)',
    account INT COMMENT 'Account of the partner'
)
COMMENT 'Table containing detailed transaction records for each account.'