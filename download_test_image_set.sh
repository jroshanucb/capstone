#Download full test images zip file from gdrive
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies  'https://docs.google.com/uc?export=download&id=1RRfK4wMBpvjAiyP_gjL2xOAfhOmXOcYO' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RRfK4wMBpvjAiyP_gjL2xOAfhOmXOcYO" -O yolo_splits4.2.zip && rm -rf /tmp/cookies.txt
unzip yolo_splits4.2.zip
