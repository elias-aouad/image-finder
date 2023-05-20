wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1g9GmYfEwSEgRcYv3GsjFH3aJFXquLgRi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g9GmYfEwSEgRcYv3GsjFH3aJFXquLgRi" -O features.zip && rm -rf /tmp/cookies.txt
unzip features.zip
rm -rf features.zip
