# Meeting Transcript — 2026-03-23
**Type:** Team Technical Discussion
**Participants:**
- Speaker 1 (Female) — Team member
- Speaker 2 (Male 1) — Primary explainer (likely Ruchit / team lead)
- Speaker 3 (Male 2) — Ran the models, receiving feedback

---

## Full Transcript

### [00:00 - 00:44] Initial Discussion on Feature Selection
**Speaker 1:** Methodology dekhna hoga na. Ek toh Ruchit ne jo features kiye the usme saare features lene hai ya fir jaise ek selected features liye the waise lene hai?
**Speaker 2:** Saare lene honge. Kyunki sir yahi bol rahe... Sir wahi bole na ki tum generalize nahi kar sakte kyunki har speech type ka ya har language ka apna apna feature hota hai dataset ka. Humne yahi toh galti ki na ki isne 9 le liya, maine top 50 le liye. But woh toh dono ke liye generalize ho gaya na. Sir toh yahi bol rahe hai ki dono ke individual lo ya fir saare parameters lo and us pe model banao.
**Speaker 1:** Okay.

### [00:44 - 00:59] Speaker 1 Steps Away
**Speaker 1:** Mujhe 10-15 minute... tum log continue karo, mujhe 10 minute do main breakfast karke aati hu.
**Speaker 2:** Okay.

### [01:00 - 02:30] Discussing 100% Accuracy and Overfitting
**Speaker 3:** Mere ko aaya kya?
**Speaker 2:** Pata hai, due to some reason or other thing, maine jab ye static symbol use kiya tha dono data ke har combination se... something has fucked up already. But 100% accuracy aayi.
**Speaker 3:** But overfitting nahi hai us case mein?
**Speaker 2:** Arre the problem is kyunki cross-validation bhi maine kar liya usme K-fold wala and testing wala bhi kar liya. Dono pe 100 aaya hai. Can you imagine? Kyunki agar kahin pe bhi overfitting ya kuch hota na toh cross-validation kharab de deta mere ko results. Ab main soch raha hu main kya bolu iske upar. Ye wala result na us din jab hamara sir ka meet tha uske pehle hi aa gaya tha. But mere ko thoda bolne mein sharam aa raha tha ki 100%...
**Speaker 3:** *(Laughs)*
**Speaker 2:** Bhai already bahut galtiyan nikal chuke the sir. 100% accuracy bolta na, sir thok dete seedhe aake.

### [02:31 - 04:18] Figuring Out the Next Steps for Feature Extraction
**Speaker 2:** Hmm, aur koi option toh nahi hai. Ya toh ek kaam kar sakte hai, we both can divide the work. Tu saara feature extraction wala part kar le...
**Speaker 3:** Matlab saare hi lene hai toh fir customize bhi kya hi hai fir toh ek hi dataset...
**Speaker 2:** Arre ek sir ne woh paper bheja hai na padhne ke liye...
**Speaker 3:** Kaunsa wala CSV?
**Speaker 2:** Accha woh wala na, right. Usi ke saare features us pe toh kar hi liya na, mere results toh usi ke upar the. Basically mere ko samajh mein nahi aa raha ab aage karna kya hai. The problem is sir bol rahe hai ki 40-40 features individually nahi le sakte. Toh mera question wahi pe tha ki bhai jab sir maine 87, 77 ke 77 leke baitha tha, aage kya karu ab?

### [04:19 - 07:30] Explaining the Flaw in Generalizing Features Across Datasets
**Speaker 2:** Arre basically sir bol rahe the na ki dono dataset jaise tune Vowel A ka liya, right? Vowel A mein bhi PC Gita ke apne features honge kyunki woh alag language mein bhi hai, aur iske alag honge Voiced wale ke. Features same hai but top features jo honge woh toh alag honge na. Toh basically tu 77 features hai hamare paas uske. Tu le raha hai top 40 ya 9, kya le raha hai? 9 le raha hai na? Dono option hai tere paas. Right, ab tu chahe 9 le ya 40 le, you are leaving the rest of the features, right? Toh tere ko check karna padega ki jo top 40 features hai, chahe woh Voiced wale ke ho ya iske ho, woh exactly same hai kya.
**Speaker 3:** *(Agrees)*
**Speaker 2:** Acche results nahi... So basically what sir meant was jaise dataset A hai, right? Uske top features 9 honge. Tere ko common nahi nikalna. Toh individually rank list nikalte hai na hum? Just find out the rank list. Right? And then compare it ki top 10 aur top 10 iska same hai kya? Ya top 40 aur top 40 same hai kya? Then you can go with this methodology ki 40 ke 40 features same hai toh maine same model rakha. Basically jo tune abhi architecture mein likha na ki 44 features... jaise maine liya K-best mein 50 features, right? So you need to give a justification ki woh 50 same hai. So there is no difference between both the datasets. Agar difference hai, toh better option would be top 10 le dono ka and merge kar de.

### [07:31 - 09:30] The Solution: Merging Top Features
**Speaker 2:** Toh merge... basically tu kya kar raha hai na, top 9 liya. Tune dono mein se nikala ki dono mein top mein similar kaun hai. Shayad 12 mein se 9 similar honge toh tune top 9 bana liya, right? But what you should do is top 9 PC Gita ka le, top 9 Voiced ka le. Theek hai? Maante hai 9 mein se 8 same hai dono ke. Toh 8 toh tera same ho gaya. Dono mein ek-ek feature shayad different hai, right? Toh un dono ko add kar le. And say ki top 9 liya and there was one differing so maine ab 10... top 9 mein se 10 ka bana liya list. Aur hum do different methodology pe ab kaam nahi kar rahe nahi toh sir hamari jaan le lenge. Woh matrix khol.

### [09:31 - 11:30] Analyzing the Confusion Matrix and Screen Sharing Issue
**Speaker 2:** Basically sir tere ko ye samjhane ki koshish kar rahe the ki kyunki... ye wala zoom kar, upar wala matrix bas. Band kyun kiya? Woh band ho gaya. Minimize nahi kar sakta, woh side tab pe jaake tu minimize karta tab nahi hoga. Tu same screen ko minimize karega toh woh band ho jata hai screen sharing. Google Meet ka naya feature hai.
**Speaker 2:** Sir basically tere ko kya bol rahe the ki jaise yahan pe jab tune combined testing kiya toh you are getting 100% accuracy, right? But the problem... ye PC Gita mein PC Gita ka training kiya toh tere ko accuracy kam aayi jabki sabse zyada aani chahiye kyunki same dataset mein training testing kar raha hai. Tere ko Voiced wale mein kam aa raha hai, that is understandable kyunki uske features same nahi honge, right? This happened because kyunki tune dono features ko combine kar diya. Toh dono ke mixed datasets ke features gaye. Dono ke individually 40 nahi gaye tere paas. So you should make separate models. PC Gita vs PC Gita ka ek alag model banega. Toh usme training testing ke liye uske liye alag features honge. Abhi tune kya kiya ek model banaya usi mein sab kuch test kar diya, right?

### [11:31 - 14:18] Diagonal True Positives and Finalizing the Plan
**Speaker 2:** And same goes to ye wala, upar aa ek baar matrix mein. Ab isme bhi yahi hua. Voiced wale mein tune kya kiya ki Voiced se Voiced testing kiya... toh is se kya samajh mein aa raha hai ki maximum features iske gaye hai, Voiced ke gaye hai. 2x2 ka tu confusion matrix nikal. Toh ye wala hoga na... first row ka do column aur second row ka two columns, right? Iska true positive ye hoga, iska true positive ye hoga. Diagonal wale dono left to right wale true positives ho jayenge dono ke liye, right? Isme problem kya ho raha hai ki tera... is se tu samajh ja ki tere zyada tar jo features gaye hai na training module mein, woh Voiced wale ke gaye hai rather than going for PC Gita, toh equal split kabhi nahi hua.
**Speaker 2:** So best approach would be ki hum dono ka rank wise dataset nikale and then we can take top 40 ya top 50 from both of them and usko merge kar de. Toh jo similar rahenge usko ek rakh denge and jo different rahenge woh add up ho jayenge uske upar. So woh karne se tera woh jo feature extraction mein jo problem ho raha hai woh bhi solve hoga. Second would be ki tera jitna bhi features missing ho rahe dono case mein toh tera matrix sahi ho jayega udhar.

### [14:19 - 15:00] Dividing the Work
**Speaker 2:** Toh haan bata tu kaunsa part karega isme, main kaunsa karu?
**Speaker 3:** Ranking ka features...
**Speaker 2:** Okay. Toh tu feature rank karke club karke dega mere ko ki clubbing main kar du?
**Speaker 3:** Haan mere ko dono de dena so I'll match it up even. Ek extracted wala de dena aur ek woh wala club wala de dena mere ko.

### [15:01 - 16:00] Speaker 1 Returns & Casual Banter
**Speaker 1:** Kya chalu hai?
**Speaker 2:** Isko... ye jaake JP Europe wala London wala teeno project tu sambhal le. Ho gaya iska toh. Haan teeno same hi hai bas ek India hai ek Global hai bas utna hi difference hai.
**Speaker 1:** Isme kya decide kiya?
**Speaker 2:** Ki Mridul ko koi kaam nahi dena hai. Usne hamari izzat ki bhaji pala kar di. Excel dekh ke kya hi bole koi yaar. Bhai humne aisa code hi nahi likha hai jisme 23 decimal uska woh ho normalization. At the end sir bhi wahan pe samajh nahi rahe the matlab sir bhi kya hi bole.

### [16:01 - 17:56] Wrapping Up and AI Meeting Recorders
**Speaker 1:** Tum log ka chal kya raha hai, tum log internal decision pehle theek se lo. Ye wale meeting ke notes banaye ya nahi?
**Speaker 3:** Nahi.
**Speaker 1:** Mere ko bhi bhej dena bhai.
**Speaker 2:** Bhai isme main kuch nahi kar sakta maine subah mail bheja usme tha kya? Dr. Preet... mujhe toh Dr. Preet ka kuch dikha hi nahi mujhe toh commercial manager ka dikha.
**Speaker 1:** Accha us din Fireflies meeting mein dikha raha tha pata nahi kyu record hua nahi.
**Speaker 2:** Haan pure time dikha raha tha woh. Bhai 340 rupaye lage hai per month. Maine uska kaat karke ab uska liya hai Read.ai ka liya hai.
