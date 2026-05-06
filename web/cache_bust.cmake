file(READ "${HTML}" html)
string(REGEX REPLACE "src=(\"index\\.js\"|index\\.js)" "src=\"index.js?v=${VERSION}\"" html "${html}")
file(WRITE "${HTML}" "${html}")
