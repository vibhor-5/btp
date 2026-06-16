import pandas as pd
import numpy as np

np.random.seed(42)
n_rows = 10000

print("Generating synthetic DataCoSupplyChainDataset.csv...")

data = {
    'Type': np.random.choice(['DEBIT', 'TRANSFER', 'CASH', 'PAYMENT'], n_rows),
    'Days for shipping (real)': np.random.randint(1, 7, n_rows),
    'Days for shipment (scheduled)': np.random.randint(1, 7, n_rows),
    'Benefit per order': np.random.uniform(-100, 300, n_rows),
    'Sales per customer': np.random.uniform(10, 500, n_rows),
    'Delivery Status': np.random.choice(['Advance shipping', 'Late delivery', 'Shipping on time', 'Shipping canceled'], n_rows),
    'Late_delivery_risk': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    'Category Id': np.random.randint(1, 50, n_rows),
    'Category Name': np.random.choice(['Sporting Goods', 'Cleats', 'Women\'s Apparel', 'Electronics', 'Men\'s Footwear'], n_rows),
    'Customer City': np.random.choice(['Caguas', 'San Jose', 'Los Angeles', 'Miami'], n_rows),
    'Customer Country': np.random.choice(['Puerto Rico', 'EE. UU.'], n_rows),
    'Customer Email': ['customer@email.com'] * n_rows,
    'Customer Fname': ['John'] * n_rows,
    'Customer Id': np.random.randint(1, 10000, n_rows),
    'Customer Lname': ['Smith'] * n_rows,
    'Customer Password': ['XXXXXXXXX'] * n_rows,
    'Customer Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], n_rows),
    'Customer State': np.random.choice(['PR', 'CA', 'NY', 'FL'], n_rows),
    'Customer Street': ['123 Main St'] * n_rows,
    'Customer Zipcode': np.random.randint(10000, 99999, n_rows),
    'Department Id': np.random.randint(1, 10, n_rows),
    'Department Name': np.random.choice(['Fitness', 'Apparel', 'Golf', 'Footwear'], n_rows),
    'Latitude': np.random.uniform(18.0, 45.0, n_rows),
    'Longitude': np.random.uniform(-120.0, -66.0, n_rows),
    'Market': np.random.choice(['Pacific Asia', 'USCA', 'Europe', 'Africa', 'LATAM'], n_rows),
    'Order City': np.random.choice(['Bikaner', 'Townsville', 'Toowoomba'], n_rows),
    'Order Country': np.random.choice(['India', 'Australia', 'China'], n_rows),
    'Order Customer Id': np.random.randint(1, 10000, n_rows),
    'order date (DateOrders)': pd.date_range(start='1/1/2015', periods=n_rows, freq='h'),
    'Order Id': np.arange(1, n_rows + 1),
    'Order Item Cardprod Id': np.random.randint(1, 100, n_rows),
    'Order Item Discount': np.random.uniform(0, 50, n_rows),
    'Order Item Discount Rate': np.random.uniform(0.0, 0.25, n_rows),
    'Order Item Id': np.arange(1, n_rows + 1),
    'Order Item Product Price': np.random.uniform(10, 500, n_rows),
    'Order Item Profit Ratio': np.random.uniform(-1.0, 0.5, n_rows),
    'Order Item Quantity': np.random.randint(1, 6, n_rows),
    'Sales': np.random.uniform(10, 1000, n_rows),
    'Order Item Total': np.random.uniform(10, 1000, n_rows),
    'Order Profit Per Order': np.random.uniform(-100, 300, n_rows),
    'Order Region': np.random.choice(['South Asia', 'Oceania', 'Eastern Asia'], n_rows),
    'Order State': np.random.choice(['Rajasthan', 'Queensland'], n_rows),
    'Order Status': np.random.choice(['COMPLETE', 'PENDING', 'CLOSED', 'PROCESSING'], n_rows),
    'Order Zipcode': np.random.randint(10000, 99999, n_rows),
    'Product Card Id': np.random.randint(1, 50, n_rows),
    'Product Category Id': np.random.randint(1, 50, n_rows),
    'Product Description': ['Description'] * n_rows,
    'Product Image': ['http://image.com/1'] * n_rows,
    'Product Name': np.random.choice(['Perfect Fitness Perfect Rip Deck', 'Nike Men\'s CJ Elite 2 TD Cleats'], n_rows),
    'Product Price': np.random.uniform(10, 500, n_rows),
    'Product Status': np.random.choice([0, 1], n_rows),
    'shipping date (DateOrders)': pd.date_range(start='1/5/2015', periods=n_rows, freq='h'),
    'Shipping Mode': np.random.choice(['Standard Class', 'First Class', 'Second Class'], n_rows)
}

df = pd.DataFrame(data)
# Add some stockout proxy dynamics
df.loc[df['Order Status'] == 'PENDING', 'Order Item Quantity'] = 0

df.to_csv('DataCoSupplyChainDataset.csv', index=False)
print("Saved to DataCoSupplyChainDataset.csv")
