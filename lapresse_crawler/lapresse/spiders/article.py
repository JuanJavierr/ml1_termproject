import scrapy
import datetime


def get_element_from_url(url, index):
    list = url.split("/")
    try:
        return list[index]
    except IndexError:
        return None


class ArticleSpider(scrapy.Spider):
    name = "article"
    allowed_domains = ["lapresse.ca"]

    def start_requests(self):
        today = datetime.datetime.today() - datetime.timedelta(days=248)
        last_year = datetime.datetime.today() - datetime.timedelta(days=279)

        delta = today - last_year
        for i in range(delta.days):
            day = last_year + datetime.timedelta(days=i)
            yield scrapy.Request(
                f"https://www.lapresse.ca/archives/{day.year}/{day.month}/{day.day}.php",
                callback=self.parse,
            )

    def parse(self, response):
        article_links = response.css(".storyTextList__itemLink::attr(href)")

        yield from response.follow_all(article_links, callback=self.save_article)

    def save_article(self, response):
        date = response.css(
            ".publicationsDate--type-publication ::attr(datetime)"
        ).get()[:10]

        yield {
            "url": response.url,
            "date": date,
            "author": response.css(".authorModule__name ::text").get(),
            "section_1": get_element_from_url(response.url, 3),
            "section_2": get_element_from_url(response.url, 4),
            "url_date": get_element_from_url(response.url, 5),
            "dossier": get_element_from_url(response.url, 6),
            "title": response.css(".titleModule__main ::text").get(),
            "subject": response.css(".titleModule__sup ::text").get(),
            "text": "\n".join(
                response.css(".articleBody > .textModule ::text").getall()
            ),
        }
