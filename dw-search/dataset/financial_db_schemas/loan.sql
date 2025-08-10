CREATE TABLE IF NOT EXISTS loan (
    loan_id INT COMMENT 'The ID number identifying the loan data',
    account_id INT COMMENT 'The ID number identifying the account',
    date DATE COMMENT 'The date when the loan was approved',
    amount INT COMMENT 'Approved loan amount in US dollars',
    duration INT COMMENT 'Loan duration in months',
    payments DOUBLE COMMENT 'Monthly payment amount',
    status STRING COMMENT 'Repayment status. A: contract finished, OK; B: contract finished, loan not paid; C: running contract, OK so far; D: running contract, client in debt.'
)
COMMENT 'Table containing loan information for each account.'