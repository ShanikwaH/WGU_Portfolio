--Requirement 1. Discuss how database design and indexing strategy optimize performance
--Requirement 2.  Describe the technical environment used in your database implementation
	--Normalization 1N to 3N (ensure every column is dependent on primary key, eliminate transitive dependencies)
	--primary indexes (primary key Order_ID)
	--single column index (Country, Item Type, and Region)
--Requirement 3. Demonstrate the functionality of the queries in the lab environment
--Requirement 4 will be discussed after the queries

CREATE TABLE Sales ( 
	SalesID SERIAL PRIMARY KEY,
	OrderID BIGINT NOT NULL,
	ItemType VARCHAR(50) NOT NULL,
	Units_Sold INT,
	Total_Revenue DECIMAL(20, 2),
	Total_Cost DECIMAL(20, 2),
	Total_Profit DECIMAL(20, 2),
	FOREIGN KEY (OrderID) REFERENCES Orders(OrderID),
	FOREIGN KEY (ItemType) REFERENCES Products(ItemType)
);

--Importing the data --
COPY Products(ItemType, UnitPrice, UnitCost)
FROM 'C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D597/Task 1'
DELIMITER ',' CSV HEADER;

COPY Orders(OrderID, OrderDate, ShipDate, OrderPriority, SalesChannel, Country, Region)
FROM 'C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D597/Task 1'
DELIMITER ',' CSV HEADER;

COPY Sales(OrderID, ItemType, UnitsSold, TotalRevenue, TotalCost, TotalProfit)
FROM 'C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D597/Task 1'
DELIMITER ',' CSV HEADER;

--Checking the import --
SELECT * FROM Sales;
SELECT DISTINCT Region FROM Orders;
SELECT DISTINCT Country FROM Orders;
SELECT DISTINCT ItemType FROM Products;

CREATE TABLE Orders (
    OrderID BIGINT PRIMARY KEY,
    OrderDate DATE NOT NULL,
    ShipDate DATE,
    OrderPriority CHAR(1),
    SalesChannel VARCHAR(20),
    Country VARCHAR(100),
    Region VARCHAR(100)
);

CREATE TABLE Products (
    ItemType VARCHAR(50) PRIMARY KEY,
    UnitPrice DECIMAL(10,2),
    UnitCost DECIMAL(10,2)
);

CREATE TABLE Sales (
    SalesID SERIAL PRIMARY KEY,
    OrderID BIGINT REFERENCES Orders(OrderID),
    ItemType VARCHAR(50) REFERENCES Products(ItemType),
    UnitsSold INT,
    TotalRevenue DECIMAL(15,2),
    TotalCost DECIMAL(15,2),
    TotalProfit DECIMAL(15,2)
);


INSERT INTO Regions (Region_Name)
SELECT DISTINCT Region
FROM Orders;

INSERT INTO Countries (Country_Name)
SELECT DISTINCT Country
FROM Orders;

INSERT INTO Item_Types (Item_Type_Name)
SELECT DISTINCT ItemType
FROM Products;

--Check Sales to see everything loaded properly--

SELECT * FROM Sales
	
--Requirement 4. Discuss how the queries solve the identified business problem--
	--Business problem: Need a flexible, scalable database, and optimization--
	
--Three business queries --
--1. Retrieve Top-Selling Product Categories --
SELECT ItemType, SUM(UnitsSold) AS TotalUnitsSold
FROM Sales
GROUP BY ItemType
ORDER BY TotalUnitsSold DESC;
	
--2. Generate a Financial Summary --
SELECT SUM(TotalRevenue) AS TotalRevenue, SUM(TotalProfit) AS NetProfit 
FROM Sales;

--3. Identify High-Priority Orders --
SELECT OrderID, OrderDate, OrderPriority 
FROM Orders 
WHERE OrderPriority = 'H' 
ORDER BY OrderDate DESC;
