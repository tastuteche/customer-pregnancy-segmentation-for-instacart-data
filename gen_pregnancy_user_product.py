import pandas as pd
import numpy as np
from tabulate import tabulate
from itertools import combinations

b_dir = '../instacart/order_products__prior.csv/'
b1_dir = '../instacart/products.csv/'
b2_dir = '../instacart/orders.csv/'

order_products__prior = pd.read_csv(b_dir + 'order_products__prior.csv')
products = pd.read_csv(b1_dir + 'products.csv')
dic = products.set_index('product_id')['product_name'].to_dict()


def get_product_set(name):
    return set(products[products['product_name'].str.lower(
    ).str.contains(name)]['product_id'].tolist())


pregnancy_set = get_product_set('pregnancy')


orders = pd.read_csv(b2_dir + 'orders.csv')
orders = orders.loc[orders['eval_set'] == 'prior']
order_prior = pd.merge(orders, order_products__prior, on=["order_id"])

op = order_prior

pregnancy_order_set = set(
    op[op.product_id.isin(pregnancy_set)]['order_id'].unique())


pregnancy_user_product = op[op.order_id.isin(pregnancy_order_set)]

pregnancy_user_product.to_csv('pregnancy_user_product.csv')
