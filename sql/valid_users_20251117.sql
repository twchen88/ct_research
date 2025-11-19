SELECT c.id, c.user_id, c.created_date, c.subscription_status FROM ct_customer.customers AS c
JOIN constant_therapy.users u
    ON u.id = c.user_id
    AND NOT u.is_demo
WHERE c.created_date > '2020-04-01 00:00:00'
	AND c.usertype = 'patient';