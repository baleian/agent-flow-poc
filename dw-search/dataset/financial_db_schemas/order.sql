CREATE TABLE IF NOT EXISTS order (
    order_id INT COMMENT 'The unique ID for the order',
    account_id INT COMMENT 'ID number of the account initiating the payment',
    bank_to STRING COMMENT 'Bank of the recipient (two-letter code)',
    account_to INT COMMENT 'Account number of the recipient',
    amount DOUBLE COMMENT 'Debited amount',
    k_symbol STRING COMMENT 'Characterization (purpose) of the payment. "POJISTNE": insurance, "SIPO": household, "LEASING": leasing, "UVER": loan payment.'
)
COMMENT 'Table containing payment order information.'