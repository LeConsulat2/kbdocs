import requests
import urllib.parse
from bs4 import BeautifulSoup


def google_translate(original_text, to_lang, from_lang="auto"):
    """
    🌐 구글 번역 함수
    Args:
        original_text (str): 번역할 텍스트
        to_lang (str): 목표 언어 (예: 'ko', 'en', 'ja')
        from_lang (str): 출발 언어 (기본값: 'auto')
    Returns:
        str: 번역된 텍스트
    """
    # 🔄 URL 인코딩
    encoded_text = urllib.parse.quote(original_text)

    # 🌍 구글 번역 모바일 URL 생성
    url = f"https://translate.google.com/m?sl={from_lang}&tl={to_lang}&q={encoded_text}"

    # 🤖 User-Agent 설정
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Mobile Safari/537.36"
    }

    # 📡 HTTP 요청
    response = requests.get(url, headers=headers)

    # 🔍 결과 추출
    soup = BeautifulSoup(response.text, "html.parser")
    result = soup.find("div", {"class": "result-container"})

    return result.text if result else ""


# 📝 사용 예제
if __name__ == "__main__":
    # 한국어 -> 영어
    result1 = google_translate("안녕하세요", "en", "ko")
    print(f"한->영: {result1}")

    # 영어 -> 한국어
    result2 = google_translate("Hello", "ko", "en")
    print(f"영->한: {result2}")
