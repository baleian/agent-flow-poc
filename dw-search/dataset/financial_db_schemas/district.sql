CREATE TABLE IF NOT EXISTS district (
    district_id INT COMMENT 'Location of the branch (unique ID)',
    A2 STRING COMMENT 'Name of the district',
    A3 STRING COMMENT 'Region where the district is located',
    A4 STRING COMMENT 'Number of inhabitants',
    A5 STRING COMMENT 'Number of municipalities with inhabitants < 499',
    A6 STRING COMMENT 'Number of municipalities with inhabitants 500-1999',
    A7 STRING COMMENT 'Number of municipalities with inhabitants 2000-9999',
    A8 STRING COMMENT 'Number of municipalities with inhabitants > 10000',
    A9 INT COMMENT 'Not useful',
    A10 DOUBLE COMMENT 'Ratio of urban inhabitants',
    A11 INT COMMENT 'Average salary in the district',
    A12 DOUBLE COMMENT 'Unemployment rate in 1995',
    A13 DOUBLE COMMENT 'Unemployment rate in 1996',
    A14 INT COMMENT 'Number of entrepreneurs per 1000 inhabitants',
    A15 INT COMMENT 'Number of committed crimes in 1995',
    A16 INT COMMENT 'Number of committed crimes in 1996'
)
COMMENT 'Table containing demographic and economic statistics for each district.'