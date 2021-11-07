mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
[theme]\n\
base=\"light\"\n\
primaryColor=\"#3FC9DA\"\n\
textColor=\"#000000\"\n\
\n\
" > ~/.streamlit/config.toml