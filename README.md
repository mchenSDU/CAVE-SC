# CAVE-SC
The source codes for learning representations of venues and categories from check-in sequences based on CAVE-SC.
The input data format is [venueID_1#category_1, ..., venueID_n#category_n]. Each line represents the check-in sequence of a user, sorted by check-in timestamp.

For example:
Input content (a partial check-ins extracted from the data):
4d8dd576c3d04eb9b829cdc9#Building,4b5b5e81f964a52062f828e3#NULL,4d8dd576c3d04eb9b829cdc9#Building,4d8dd576c3d04eb9b829cdc9#Building,4bc482934cdfc9b6fcb09821#Plaza,4ccacd52e47f5481d8f0c9f7#Office, ...
