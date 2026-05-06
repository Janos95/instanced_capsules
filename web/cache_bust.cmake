file(READ "${HTML}" html)
string(REPLACE "src=\"index.js\"" "src=\"index.js?v=${VERSION}\"" html "${html}")
file(WRITE "${HTML}" "${html}")
