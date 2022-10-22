import pandas as pd
authors = pd.DataFrame({
    "author_id": [1, 2, 3],
    "author_name": ['Тургенев', 'Чехов', 'Островский']
}
)
print(authors)

book = pd.DataFrame({
    "author_id": [1, 1, 1, 2, 2, 3, 3],
    "book_title": ['Отцы и дети', 'Рудин', 'Дворянское гнездо',
                   'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
    "price": [450, 300, 350, 500, 450, 370, 290]
})

print(book)

authors_price = pd.merge(authors, book, on="author_id", how='outer')

print(authors_price)

top5 = authors_price.sort_values('price', ascending=False).head()

print(top5)

top5_1 = authors_price.nlargest(5, 'price').reset_index(drop=True)
print(top5_1)

authors_stat = authors_price.groupby('author_name').agg({'price':['min', 'max', 'mean']})

authors_stat = authors_stat.rename(columns={'min':'min_price', 'max':'max_price', 'mean':'mean_price'})

print(authors_stat)