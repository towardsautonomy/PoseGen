from icrawler.builtin import GoogleImageCrawler

# google_crawler = GoogleImageCrawler(
#     feeder_threads=1,
#     parser_threads=2,
#     downloader_threads=8,
#     storage={'root_dir': 'data/tesla_model_3'})
# filters = dict(
#     size='large',
#     color='black',
#     license='commercial,modify',
#     date=((2020, 1, 1), (2021, 9, 30)))
# google_crawler.crawl(keyword='Tesla Model 3', filters=filters, max_num=100, file_idx_offset=0)

# google_crawler = GoogleImageCrawler(
#     feeder_threads=1,
#     parser_threads=2,
#     downloader_threads=8,
#     storage={'root_dir': 'data/mustang_gt_orange'})
# filters = dict(
#     size='large',
#     color='orange',
#     license='commercial,modify',
#     date=((2017, 1, 1), (2018, 12, 30)))
# google_crawler.crawl(keyword='Mustang GT', filters=filters, max_num=100, file_idx_offset=0)

# google_crawler = GoogleImageCrawler(
#     feeder_threads=1,
#     parser_threads=2,
#     downloader_threads=8,
#     storage={'root_dir': 'data/bmw_328i_black'})
# filters = dict(
#     size='large',
#     color='black',
#     license='commercial,modify',
#     date=((2014, 1, 1), (2021, 12, 30)))
# google_crawler.crawl(keyword='BMW 328i', filters=filters, max_num=100, file_idx_offset=0)

# google_crawler = GoogleImageCrawler(
#     feeder_threads=1,
#     parser_threads=2,
#     downloader_threads=8,
#     storage={'root_dir': 'data/ford_f150_black'})
# filters = dict(
#     size='large',
#     color='black',
#     license='commercial,modify',
#     date=((2014, 1, 1), (2021, 12, 30)))
# google_crawler.crawl(keyword='Ford F150', filters=filters, max_num=100, file_idx_offset=0)

# google_crawler = GoogleImageCrawler(
#     feeder_threads=1,
#     parser_threads=2,
#     downloader_threads=8,
#     storage={'root_dir': 'data/ford_f150_white'})
# filters = dict(
#     size='large',
#     color='white',
#     license='commercial,modify',
#     date=((2014, 1, 1), (2021, 12, 30)))
# google_crawler.crawl(keyword='F-150', filters=filters, max_num=100, file_idx_offset=0)

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=2,
    downloader_threads=8,
    storage={'root_dir': 'data/tesla_cybertruck'})
filters = dict(
    size='large',
    license='commercial,modify',
    date=((2014, 1, 1), (2021, 12, 30)))
google_crawler.crawl(keyword='Tesla Cybertruck', filters=filters, max_num=100, file_idx_offset=0)