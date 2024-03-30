/*Use the products table from SQL_STORE and write a query to display the name, 
unit price, and a new price with a 10% increase rate.*/
use sql_store;

select p.name, p.unit_price, round((p.unit_price * 0.1 + p.unit_price), 2) as 'new_unit_price' 
from products as p;






select p.name, p.unit_price, truncate((p.unit_price * 0.1 + p.unit_price), 2) as 'new_unit_price' 
from products as p;





/*Write a query to get the items for order #6 with a total price of over 30. Use the 
ORDER_ITEMS table from SQL_STORE database*/

select o.order_id, p.product_id, p.name, o.quantity, o.unit_price, 
(o.quantity * o.unit_price) as 'total_price'
from order_items as o inner join products as p on o.product_id = p.product_id
where o.order_id = 6 and (o.quantity * o.unit_price) > 30;





select *, (quantity * unit_price) as 'total_price' from order_items 
where order_id = 6 and (quantity * unit_price) > 30;


select product_id from order_items 
where order_id = 6 and (quantity * unit_price) > 30;



select p.name as 'Product Name'
from order_items as o inner join products as p on o.product_id = p.product_id
where o.order_id = 6 and (o.quantity * o.unit_price) > 30;




/*Use SQL_STORE database & join the ORDER_ITEMS table with the products table 
on the PRODUCT_ID column and select the specific columns of your choice. */

select p.name as 'Product', o.order_id as 'Order Id', o.quantity as 'Order Qty', 
o.unit_price as 'Price', p.unit_price as 'Cost', p.quantity_in_stock as 'Stock Qty'  
from order_items as o inner join products as p on o.product_id = p.product_id;






/*Join products table of SQL_INVENTORY with ORDER_ITEM_NOTES table of 
SQL_STORE on PRODUCT_ID and select specific columns*/



select p.name as 'product_name', p.quantity_in_stock, p.unit_price, o.note 
from sql_inventory.products as p inner join sql_store.order_item_notes as o
on p.product_id = o.product_id;







/*Use the ‘SQL_INVOICING’ database. Join the ‘payments’ table with the ‘clients’ 
table using the ‘CLINETS_ID’ column and then join the ‘payments’ table with the 
‘PAYMENT_METHODS’ table using the ‘PAYMENT_METHOD’ column.*/

use sql_invoicing;


select * from payments as p inner join clients as c on
p.client_id = c.client_id inner join payment_methods as m on 
m.payment_method_id = p.payment_method;





select p.date, p.amount, c.name, c.address, c.city, c.state, c.phone,
m.name as 'payment_method'
from payments as p inner join clients as c on
p.client_id = c.client_id inner join payment_methods as m on 
m.payment_method_id = p.payment_method;






/*Write a query to perform left join on orders table and shippers table 
of SQL_STORE on SHIPPER_ID and get the details of ORDER_ID, CUSTOMER_ID, 
ORDER_DATE, SHIPPED_DATE & SHIPPERS_NAME*/


select o.order_id, o.customer_id, o.order_date, o.shipped_date 
from sql_store.orders as o left outer join sql_store.shippers as s 
on o.shipper_id = s.shipper_id;




/*Create a database with the name MYDBMS and create a ‘PERSONS’ table under 
MYDBMS database with the following columns. 
PERSON_ID, 
FIRSTNAME, 
LASTNAME, 
AGE. 
Make ‘PERSON_ID’ as primary key and give the appropriate data type to each column.*/






select * from sql_invoicing.payments as p inner join sql_invoicing.clients as c on
p.client_id = c.client_id inner join sql_invoicing.payment_methods as m on 
m.payment_method_id = p.payment_method;



/*-----------------------------------------------------------------------------------*/


create database `mydbms`;
use `mydbms`;



create table `persons` (
  `person_id` int(6),
  `firstname` varchar(15),
  `lastname` varchar(25),  
  `age` int(3),
  primary key (`person_id`));

select * from persons;


create table `persons` (
  `person_id` int(6) not null auto_increment,
  `firstname` varchar(15),
  `lastname` varchar(25),  
  `age` int(3),
  primary key (`person_id`));

select * from persons;




create table `persons` (
  `person_id` int not null auto_increment,
  `firstname` varchar(15) default null,
  `lastname` varchar(25) default null,
  `age` int default null,
  primary key (`person_id`)
); 


select * from persons;


INSERT INTO `persons` VALUES (1,'Sachin','Tendulkar',50);
INSERT INTO `persons` VALUES (2,'Taimur','Khan',10);
INSERT INTO `persons` VALUES (3,'Brad','Pitt',55);
INSERT INTO `persons` VALUES (4,'John','Wick',53);
INSERT INTO `persons` VALUES (5,'James','Bond',47);
INSERT INTO `persons` VALUES (6,'Yuraj','Singh',43);




INSERT INTO persons (person_id,firstname, lastname, age) VALUES (1,'Sachin','Tendulkar',50);
INSERT INTO persons (person_id,firstname, lastname, age) VALUES (2,'Taimur','Khan',10);
INSERT INTO persons (person_id,firstname, lastname, age) VALUES (3,'Brad','Pitt',55);
INSERT INTO persons (person_id,firstname, lastname, age) VALUES (4,'John','Wick',53);
INSERT INTO persons (person_id,firstname, lastname, age) VALUES (5,'James','Bond',47);
INSERT INTO persons (person_id,firstname, lastname, age) VALUES (6,'Yuraj','Singh',43);

select * from persons;




/*Use the ‘SQL_INVOICING’ database. 
Join the ‘payments’ table with the ‘clients’ table using the ‘CLINETS_ID’ 
column and then join the ‘payments’ table with the ‘PAYMENT_METHODS’ table using 
the ‘PAYMENT_METHOD’ column. Select the following columns.
PAYMENT_ID from the payments table,
Amount from the payments table,
CLIENT_ID from the clients table,
Name as CLINET_NAME from the clients table,
Phone as CLIENT_PHONE from the clients table,
Name as PAYMENT_METHOD from PAYMENT_METHODS table
*/

use sql_invoicing;

select p.payment_id,  p.amount, c.client_id, c.name as 'CLINET_NAME', c.phone as 'CLIENT_PHONE',
pm.name as 'PAYMENT_METHOD'
from payments as p inner join clients as c on p.client_id = c.client_id
inner join payment_methods as pm on pm.payment_method_id = p.payment_method;




select p.payment_id,  p.amount, c.client_id, c.name as 'CLINET_NAME', c.phone as 'CLIENT_PHONE',
pm.name as 'PAYMENT_METHOD'
from 
(payment_methods as pm inner join 
 (payments as p inner join clients as c on p.client_id = c.client_id)
on pm.payment_method_id = p.payment_method); 


/*Create a ‘payment_details’ table*/

create table `payment_details` (
  `tbl_id` int(11) not null auto_increment,	
  `payment_id` int(11) not null,
  `amount` decimal(9,2) not null,
  `client_id` int(11) not null,
  `client_name` varchar(50) not null,
  `client_phone` varchar(50) default null,
  `payment_method` varchar(50) NOT NULL,
  primary key (`tbl_id`));

select * from payment_details;








insert into payment_details (payment_id, amount, client_id, client_name, client_phone, 
payment_method) 
select p.payment_id,  p.amount, c.client_id, c.name, 
c.phone, pm.name
from 
(payment_methods as pm inner join 
 (payments as p inner join clients as c on p.client_id = c.client_id)
on pm.payment_method_id = p.payment_method); 


select * from payment_details;





/*Refer ‘PERSONS’ table, modify the column constraint of AGE and LASTNAME and 
change it to not null*/


alter table persons change `age` `age` int not null;
alter table persons change `lastname` `lastname` varchar(25) not null;

/*after change the column constraints*/
/*below is the system generated sql script*/
CREATE TABLE `persons` (
  `person_id` int NOT NULL AUTO_INCREMENT,
  `firstname` varchar(15) DEFAULT NULL,
  `lastname` varchar(25) NOT NULL,
  `age` int NOT NULL,
  PRIMARY KEY (`person_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;




/*In the ‘PERSONS’ table, update the income of PERSON_ID 2, as 100 times of his age*/


alter table persons add column `income` decimal(9,2) default null;





update persons set income = (age * 100) where person_id = 2;

select * from persons;




/*Write a query to give 50 extra points to customers born before 1990. Use the customers table of SQL_STORE*/

use sql_store;

select * from customers;

select * from customers where birth_date < '1990-01-01';



update customers set points = (points + 50) where birth_date < '1990-01-01';

select * from customers;














