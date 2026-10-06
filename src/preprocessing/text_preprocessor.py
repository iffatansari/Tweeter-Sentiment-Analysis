def normalize_tweet(text: str) -> str:
    if not isinstance(text,str): return ""
    text=html_lib.unescape(text).replace("\r"," ").replace("\n"," ")
    text=re.sub(r"https?://\S+|www\.\S+"," http ",text)
    text=re.sub(r"(?<!\w)@\w+"," @user ",text)
    text=re.sub(r"\s+"," ",text).strip()
    text=re.sub(r"([!?.,])\1{4,}",r"\1\1\1",text)
    return text
