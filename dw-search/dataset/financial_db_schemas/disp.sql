CREATE TABLE IF NOT EXISTS disp (
    disp_id INT COMMENT 'Unique number identifying this row of record',
    client_id INT COMMENT 'ID number of the client',
    account_id INT COMMENT 'ID number of the account',
    type STRING COMMENT 'Type of disposition. "OWNER" has full rights, "USER": standard user with access rights, "DISPONENT" can issue orders or apply for loans.'
)
COMMENT 'Table linking clients to accounts with specific rights (dispositions).'