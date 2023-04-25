-- only patients and those with time since onset are included in this query
select ct_customer.customers.id, ct_customer.customers.condition_since, ct_customer.customers.age_group, ct_customer.customers.birth_year, ct_customer.customers_to_deficits.deficit_id, ct_customer.customers_to_disorders.disorder_id
from ct_customer.customers
inner join ct_customer.customers_to_deficits on ct_customer.customers.id = ct_customer.customers_to_deficits.customer_id
inner join ct_customer.customers_to_disorders on ct_customer.customers.id = ct_customer.customers_to_disorders.customer_id
where (ct_customer.customers.usertype = "patient") and (ct_customer.customers.condition_since is not NULL) and (ct_customer.customers.subscription_status is not NULL) and (ct_customer.customers_to_disorders.disorder_id <> "9999")
and (ct_customer.customers_to_deficits.deficit_id <> "9999")
order by id asc;