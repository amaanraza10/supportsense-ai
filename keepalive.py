"""Opens the Streamlit app in a headless browser and, if it is asleep, clicks the wake-up button.
Run by the GitHub Action in keepalive.yml. Best effort: Streamlit may change its sleep page."""
import os

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "https://YOUR-APP-NAME.streamlit.app")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, timeout=120_000)
    page.wait_for_timeout(8_000)
    wake = page.get_by_role("button", name="Yes, get this app back up!")
    if wake.count():
        wake.first.click()
        print("The app was asleep. Sent the wake-up click.")
        page.wait_for_timeout(90_000)
    else:
        print("The app was already awake.")
    browser.close()
