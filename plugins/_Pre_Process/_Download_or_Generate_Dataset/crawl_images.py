# Copyright 2024 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse

from icrawler.builtin import GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler

from nnabla import logger


def func(args):
    # Prepare keywords and folders
    keywords = args.keywords.split(',')
    if args.folders is not None:
        folders = args.folders.split(',')
        if len(keywords) != len(folders):
            logger.critical(
                f'Keyword and folder sizes are different. keywords={keywords}, folders={folders}')
            exit(1)
    else:
        folders = keywords

    # Prepare filters
    filters = {}
    if args.type is not None:
        filters['type'] = args.type
    if args.color is not None:
        filters['color'] = args.color
    if args.size is not None:
        filters['size'] = args.size
    if args.license is not None:
        filters['license'] = args.license
    if args.date is not None:
        filters['date'] = args.date

    # Search and download
    for keyword, folder in zip(keywords, folders):
        logger.log(99, f'Downloading "{keyword}" images...')
        storage = {"root_dir": os.path.join(args.output, folder)}
        if args.search_engine == 'google':
            crawler = GoogleImageCrawler(storage=storage)
        elif args.search_engine == 'bing':
            crawler = BingImageCrawler(storage=storage)
        elif args.search_engine == 'baidu':
            crawler = BaiduImageCrawler(storage=storage)
        else:
            logger.critical(
                f'Search engine {args.search_engine} is not supported.')
            exit(1)

        crawler.crawl(keyword=keyword, filters=filters,
                      max_num=args.max_images)

    logger.log(99, 'Download process completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Image Crawler\n\n' +
        'Download images using icrawler.\n' +
        'Please refer to the document about icrawler for how to set each option.\n' +
        'https://icrawler.readthedocs.io/en/latest/builtin.html#search-engine-crawlers\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output',
        help='dir for saving downloaded images (dir)',
        required=True)
    parser.add_argument(
        '-k',
        '--keywords',
        help='search keywords separated by commas (text)',
        required=True)
    parser.add_argument(
        '-f',
        '--folders',
        help='folder names separated by commas. keyword are used if blank (text)')
    parser.add_argument(
        '-m',
        '--max_images',
        help='maximum number of images per keyword (int) default=100',
        type=int,
        required=True)
    parser.add_argument(
        '-se',
        '--search_engine',
        help='search engine used for image search (option:google,bing,baidu) default=google',
        default='google')
    parser.add_argument(
        '-t',
        '--type',
        help='type options separated by commas. (text)')
    parser.add_argument(
        '-c',
        '--color',
        help='color options separated by commas. (text)')
    parser.add_argument(
        '-s',
        '--size',
        help='size options separated by commas. (text)')
    parser.add_argument(
        '-l',
        '--license',
        help='license options separated by commas. (text)')
    parser.add_argument(
        '-d',
        '--date',
        help='date options separated by commas. (text)')
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
