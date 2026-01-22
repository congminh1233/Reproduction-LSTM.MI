from training import load_lstm_mi, lstm_mi_predict

pipeline = load_lstm_mi("saved_model")

domains_with_family = [
    ("vxwohebtinngsa.co.uk", "Cryptolocker - Flashback"),
    ("nsexuhasmygjxo.co.uk", "Cryptolocker - Flashback"),

    # Post Tovar GOZ
    ("nc27iy2c09ca1qu1wm0rfbb5a.com", "Post Tovar GOZ"),
    ("15tznf01ubnu8r13xbrynk4hw99.com", "Post Tovar GOZ"),

    # geodo -> Others DGA
    ("tulevuefsphtoofc.eu", "Others DGA"),
    ("wpmsrvghiiejnvsq.eu", "Others DGA"),

    # dyre
    ("acf2370083f32f74415d218812fb3f7f4b.cc", "dyre"),
    ("a706bf30110fb3d0b31bc4e27bbe22c9b0.cc", "dyre"),

    # corebot -> Others DGA
    ("6el7t5jw03fy.ddns.net", "Others DGA"),

    # symmi -> Others DGA
    ("imraubpoos.ddns.net", "Others DGA"),

    # padcrypt -> Others DGA
    ("lbfmnfcclndkdfak.online", "Others DGA"),
    ("fdccdaneaaeoaaab.co.uk", "Others DGA"),

    # tinba
    ("jerfxuebsqlm.biz", "tinba"),

    # murofet
    ("a37gxoue41jwf12dti25hzhuevj16e11gtg23p42.biz", "murofet"),

    # ranbyus
    ("yvqyqoeibgyhwm.me", "ranbyus"),

    # locky
    ("yyopkqm.pw", "locky"),

    # Volatile Cedar / Explosive
    ("ashplayergetadolbef.net", "Volatile Cedar / Explosive"),

    # beebone -> Others DGA
    ("ns1.timechk7.net", "Others DGA"),

    # bedep -> Others DGA
    ("azdjtxfhavpwfb8z.com", "Others DGA"),

    # fobber -> Others DGA
    ("qdklotayyhtbojayk.net", "Others DGA"),

    # necurs
    ("aeerdlyfbrxexoxglrotb.mn", "necurs"),

    # qakbot
    ("clplrfgzkiwjdmksopez.info", "qakbot"),

    # tempedreve -> Others DGA
    ("zolgfmdm.info", "Others DGA"),

    # qadars
    ("5qb0xuno963c.top", "qadars"),

    # ramdo
    ("aaaacqmeeoeumwey.org", "ramdo"),

    # kraken
    ("adihitbiu.mooo.com", "kraken"),

    # bamital -> Others DGA
    ("029ce0ea9c655f4404341bfb9ac43c4b.co.cc", "Others DGA"),

    # ngioweb
    ("mdgxmjiwmjua.org", "ngioweb"),
    ("anticotusely.name", "ngioweb"),
    ("anticufobing-imocinuth-refaxafish.net", "ngioweb"),
    ("antidabupity.name", "ngioweb"),
    ("antidakozossion-inuzipiless-underinunoxity.net", "ngioweb"),
    ("imadasosion-exexizuhood.com", "ngioweb"),
    ("imafimancy-inehuten.org", "ngioweb"),
    ("imafovese.info", "ngioweb"),
    ("imagavihood.net", "ngioweb"),

    # chinad
    ("0v9zy8qkoa2f3rkd.net", "chinad"),

    # gozi -> Others DGA
    ("veniispenastollere.com", "Others DGA"),

    # sphinx -> Others DGA
    ("accyqgocwvgcnpyt.com", "Others DGA"),

    # proslikefan -> Others DGA
    ("aadcuimttjpd.name", "Others DGA"),

    # vidro -> Others DGA
    ("aciewrbjei.net", "Others DGA"),

    # madmax -> Others DGA
    ("3nxauqsdgm.org", "Others DGA"),

    # dromedan -> Others DGA
    ("14b3x6oa.ru", "Others DGA"),

    # g01 -> Others DGA
    ("asinust.doesntexist.com", "Others DGA"),

    # pandabanker -> Others DGA
    ("227cfc52de98.com", "Others DGA"),

    # Tinynuke -> Others DGA
    ("05aa0f3f57c4420a83d555ffb0117578.com", "Others DGA"),

    # mirai -> Others DGA
    ("kedbuffigfjs.online", "Others DGA"),

    # unknownjs -> Others DGA
    ("czyfrwo.eu", "Others DGA"),

    # MyDoom
    ("eawpewhess.ws", "MyDoom"),
    ("cgusuqeaueysokyu.xyz", "MyDoom"),

    # Enviserv -> Others DGA
    ("0029851f6d.org", "Others DGA"),

    # Unknown Malware family -> Others DGA
    ("albdfhln.com", "Others DGA"),

    # pitou
    ("caenoaqab.mobi", "pitou"),

    # monerodownloader
    ("004a6dcb508db.blackfriday", "monerodownloader"),

    # zloader
    ("adyddfhfhlfhfhlvorip.com", "zloader"),

    # kingminer -> Others DGA
    ("25120825fdae.tk", "Others DGA"),

    # coffeeloader
    ("144585138.com", "coffeeloader"),

    # bazzarbackdoor
    ("agahaqli.bazar", "bazarbackdoor"),

    # necro
    ("fZbyZxllyAUdhOWD.xyz", "necro"),

    # flubot
    ("aallyhbvihgiflj.cn", "flubot"),

    # infy
    ("0b7fc889.space", "infy"),

    # sharkbot
    ("b9d7b1ffee2b6200.info", "sharkbot"),

    # tufik
    ("aenoolulhusfsoor.com", "tufik"),

    # verblecon
    ("a5b02982f25e36845848431b962f1af2.tk", "verblecon"),

    # orchard
    ("00f0318d.duckdns.org", "orchard"),
    ("00f0318d.org", "orchard"),

    # ares -> Others DGA
    ("fgstucskwpinvieelxoulfau.com", "Others DGA"),

    # copperstealer
    ("21c7cccc6c458c90.xyz", "copperstealer"),

    # grandoreiro -> Others DGA
    ("iuc1tjb0sas1tan.freedynamicdns.org", "Others DGA"),

    # bumblebee
    ("00jbhxdqn2l.life", "bumblebee"),

    # vipersoftx -> Others DGA
    ("bideo-schnellvpn.com", "Others DGA"),
    ("bideo-blog.xyz", "Others DGA"),
    ("bideo-cdn.com", "Others DGA"),
    ("bideo-cdn.xyz", "Others DGA"),

    # ftcode -> Others DGA
    ("ahmwmjuxmjuw.top", "Others DGA"),

    # chimera -> Others DGA
    ("tnt69eqbib53nbj3.chimerasandbox.workers.dev", "Others DGA"),

    ("avg.co.jp", "non-dga"),
    ("noticiaacompimenta.blogspot.com.br", "non-dga"),
    ("zunox.hk", "non-dga"),
    ("fortnumandmason.com", "non-dga"),
    ("shockchan.com", "non-dga"),
    ("nationalparalegal.edu", "non-dga"),
    ("henghost.com", "non-dga"),
    ("japanxthai.com", "non-dga"),
    ("xqwh.org", "non-dga"),
    ("athina984.gr", "non-dga"),
    ("optout-xjql.net", "non-dga"),
    ("travelmole.com", "non-dga"),
    ("tanktrouble.com", "non-dga"),
    ("youyouwin.com", "non-dga"),
    ("time2post.ru", "non-dga"),
]

domains = [d[0] for d in domains_with_family]

preds = lstm_mi_predict(domains, pipeline)

for pred in preds:
    print(pred)

print(f"\n\n{'No.':4} | {'Domain':50} | {'is_DGA':7} | {'Predicted':30} | {'Expected':30} | {'Correct':7}")
print("-" * 160)
for i, ((domain, expected), pred) in enumerate(zip(domains_with_family, preds), start=1):
    predicted_family = pred["family"]
    label = pred["label"]
    correct = predicted_family == expected
    is_dga = not (label == 0)
    print(f"{i:<4} | {domain:50} | {str(is_dga):7} | {predicted_family:30} | {expected:30} | {str(correct):7}")
