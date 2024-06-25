import logging
import re
import typing

import bs4
import playwright.sync_api
import htmlmin

from . import config


class PageNotLoadedException(Exception):
    pass


JS_FUNCTIONS = """
    function setValueAsDataAttribute() {
      const inputs = document.querySelectorAll('input, textarea, select');
      inputs.forEach(input => {
        const value = input.value;
        input.setAttribute('data-current-value', value);
      });
    }

    function isVisible(element) {
        return !!(element.offsetWidth || element.offsetHeight || element.getClientRects().length);
    }

    function markInvisibleElements() {
        const allElements = document.querySelectorAll('*');
        allElements.forEach(element => {
            if (!isVisible(element)) {
                element.setAttribute('data-playwright-invisible', 'true');
            }
        });
    }

    window.setValueAsDataAttribute = setValueAsDataAttribute;
    window.markInvisibleElements = markInvisibleElements;
"""


def iterate_html(
    page: playwright.sync_api.Page, minified: bool = True
) -> typing.Iterable[str]:
    html = get_full_html(page, minified=minified)
    html_tokens = html.split("<")

    i = 0

    while i < len(html_tokens):
        part = ""
        while (
            i < len(html_tokens)
            and len(part) + len(html_tokens[i]) < config.HTML_PART_LENGTH
        ):
            part += ("<" if i > 0 else "") + html_tokens[i]
            i += 1
        yield part


def get_full_html(page: playwright.sync_api.Page, minified: bool = True) -> str:
    page.evaluate("setValueAsDataAttribute()")
    page.evaluate("markInvisibleElements()")

    for el in page.locator(":visible").all():
        try:
            el.evaluate(
                "el => el.setAttribute('data-playwright-visible', true)",
                timeout=config.PLAYWRIGHT_TIMEOUT,
            )
        except playwright.sync_api.TimeoutError:
            logging.warning("Timeout error while setting visible attribute")

    if page.url == "about:blank":
        raise PageNotLoadedException("No page loaded yet")
    html = page.content()
    soup = bs4.BeautifulSoup(html, "html.parser")
    _remove_invisible(soup)
    _remove_useless_tags(soup)
    _clean_attributes(soup)
    html_clean = soup.prettify()
    html_clean = _remove_comments(html_clean)

    if minified:
        html_clean = htmlmin.minify(html_clean)

    return html_clean


def _remove_useless_tags(soup: bs4.BeautifulSoup):
    tags_to_remove = [
        "path",
        "meta",
        "link",
        "noscript",
        "script",
        "style",
    ]
    for t in soup.find_all(tags_to_remove):
        t.decompose()


def _clean_attributes(soup: bs4.BeautifulSoup, classes: bool = True):
    allowed_attrs = [
        "id",
        "name",
        "value",
        "placeholder",
        "data-test-id",
        "data-testid",
        "data-current-value",
        "data-playwright-invisible",
        "href",
    ]

    if not classes:
        allowed_attrs.append("class")

    for element in soup.find_all(True):
        element.attrs = element.attrs = {
            key: value for key, value in element.attrs.items() if key in allowed_attrs
        }


def _remove_comments(html: str):
    return re.sub(r"[\s]*<!--[\s\S]*?-->[\s]*?", "", html)


def _remove_invisible(soup: bs4.BeautifulSoup):
    invisible_elements = soup.find_all(attrs={"data-playwright-invisible": True})
    for element in invisible_elements:
        element.decompose()
