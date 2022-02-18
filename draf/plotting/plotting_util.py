from draf.prep.data_base import SRC


def make_clickable_src(src: str) -> str:
    """Converts a src_key into a html href string"""
    try:
        url = getattr(SRC, src[1:]).url
        return f"<a href='{url}' target='_blank'>{src}</a>"
    except AttributeError:
        return src
