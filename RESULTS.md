| Metric | RNN | GRU | LSTM |
|---|---|---|---|
| Parameters | 2,071,038 | 3,647,998 | 4,436,478 |
| Training time (s) | 36 | 42 | 43 |
| Best epoch | 11 | 7 | 9 |
| Train loss (best epoch) | 2.895 | 1.712 | 1.567 |
| Val loss (best) | 3.312 | 2.761 | 2.791 |
| Test loss | 3.457 | 2.976 | 2.974 |
| Test BLEU | 2.46 | 7.41 | 7.88 |
| BLEU short (<= 5 words, n=84) | 6.19 | 8.82 | 11.42 |
| BLEU long (>= 10 words, n=57) | 2.22 | 5.49 | 7.43 |

[1] EN : I met him at the barber's.
     REF: मैं उससे नाई के यहाँ मिला ।
     RNN : मैं <UNK> <UNK> <UNK> के लिए कहा गया ।
     GRU : मैं उससे <UNK> <UNK> <UNK> <UNK> हूँ ।
     LSTM: मैं उसके <UNK> में <UNK> ।

[2] EN : My watch isn't working properly.
     REF: मेरी घड़ी ख़राब हो गई ।
     RNN : मेरी <UNK> <UNK> <UNK> हैं ।
     GRU : मेरी <UNK> <UNK> <UNK> <UNK> नहीं है ।
     LSTM: मेरी <UNK> <UNK> <UNK> है है ।

[3] EN : This road is very narrow.
     REF: यह रास्ता बहुत पतला है ।
     RNN : यह एक <UNK> <UNK> है ।
     GRU : यह <UNK> बहुत <UNK> है ।
     LSTM: यह <UNK> बहुत <UNK> है ।

[4] EN : We should love our neighbors.
     REF: हमे पड़ोसियों के साथ प्रेमपूर्वक रहना चहिए ।
     RNN : हमे <UNK> <UNK> <UNK> ।
     GRU : हमे हमे <UNK> को <UNK> ।
     LSTM: हमे <UNK> <UNK> <UNK> चाहिए ।

[5] EN : Hello!
     REF: नमस्कार ।
     RNN : <UNK> <UNK> ।
     GRU : <UNK> !
     LSTM: <UNK> !

[6] EN : It's no use arguing with him.
     REF: उससे बहस करने में कोई फ़ायदा नहीं है ।
     RNN : वह मेरी <UNK> <UNK> <UNK> ।
     GRU : कोई कोई कोई नहीं नहीं नहीं है ।
     LSTM: कोई कोई भी <UNK> नहीं नहीं है ।

[7] EN : When will you return?
     REF: तुम कब लौट कर आओगे ?
     RNN : आप कहाँ हैं ?
     GRU : तुम <UNK> <UNK> <UNK> क्या ?
     LSTM: तुम कब कब हुई ?

[8] EN : I was assaulted.
     REF: मुझपर हमला किया गया ।
     RNN : मैं <UNK> ।
     GRU : मैं <UNK> था ।
     LSTM: मैं <UNK> गया था ।

[9] EN : I felt that I should help her.
     REF: मुझे लगा कि मुझे उसकी मदद करनी चाहिए ।
     RNN : मैं तुम्हें यह बात नहीं ।
     GRU : मैं उससे पहचानता हूँ कि मैंने उसे <UNK> ।
     LSTM: मैं उसे था कि मैं उससे <UNK> ।

[10] EN : Were you told to do so?
     REF: तुम्हें यह करने के लिए बताया गया था ?
     RNN : तुम कभी भी मदद क्यों नहीं ?
     GRU : तुम्हें तुम्हें लिए करना चाहिए ?
     LSTM: तुम तुम्हें <UNK> करना चाहते हो ?

[11] EN : Please pardon me for coming late.
     REF: मुझे देर से आने के लिए माफ़ कीजिएगा ।
     RNN : तुम्हारा <UNK> <UNK> <UNK> <UNK> <UNK> ।
     GRU : मुझे अपने <UNK> के लिए लिए <UNK> ।
     LSTM: मेरी माँ के लिए <UNK> <UNK> ।

[12] EN : The rumor is true to some extent.
     REF: वह अफ़वाह किसी हद तक सच है ।
     RNN : यह मेरी पत्नी <UNK> है ।
     GRU : वह क्लास में में रहता है ।
     LSTM: वह अफ़वाह है है ।

[13] EN : Please turn on the television.
     REF: टीवी चालू कीजिए ।
     RNN : कृपया धीमी मत करो ।
     GRU : टीवी चालू कर दीजिए ।
     LSTM: बत्ती चालू कर दीजिए ।

[14] EN : You can't prove that.
     REF: तुम वह साबित नहीं कर सकते ।
     RNN : हम सब <UNK> करते हैं ।
     GRU : तुम्हें तुम्हें <UNK> नहीं नहीं ।
     LSTM: तुम्हें इतना <UNK> नहीं चाहिए ।

[15] EN : Turn your face this way.
     REF: अपना मूँह इस ओर मोड़ो ।
     RNN : अपने हाथ <UNK> <UNK> <UNK> ।
     GRU : इस समस्या को <UNK> <UNK> ।
     LSTM: इस अपने <UNK> को <UNK> ।
