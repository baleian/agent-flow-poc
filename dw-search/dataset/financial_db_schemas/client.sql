CREATE TABLE IF NOT EXISTS client (
    client_id INT COMMENT 'The unique number for the client',
    gender STRING COMMENT 'Gender of the client. "F": female, "M": male.',
    birth_date DATE COMMENT 'Birth date of the client',
    district_id INT COMMENT 'Location of the branch'
)
COMMENT 'Table containing client demographic information.'