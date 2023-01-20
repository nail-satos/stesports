# HTMLの独自短縮タグを標準のタグに変換する関数
def trans_html_tag(message):
        
        message = message.replace('<R>', '<font color=red>')
        message = message.replace('<G>', '<font color=green>')
        message = message.replace('<B>', '<font color=blue>')
        message = message.replace('<P>', '<font color=purple>')
        message = message.replace('</>', '</font>')
        message = message.replace('<C>', '<br>')

        return message

# HTML文字列を生成する関数（枠）
def make_html_frame(str_title, message):

        str1 = """
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>

        /* 枠 */
        .frame {
        position: relative;
        display: inline-block;
        /* margin: 30px 10px; */
        /* padding: 20px 10px 10px 20px; */
        margin: 10px;
        padding: 20px 10px 10px 20px;
        background-color: #EEEEEE;
        border-radius: 10px;
        }

        /* 枠 */
        .frame::before {
        content: '"""

        str2 = """';
        position: absolute;
        top: -15px;
        left: -10px;
        padding: 10px;
        /* background-color: #3232cd; */
        background-color: #8a8afa;
        border-radius: 12px;
        font-size: 16px;
        color: #fff;
        font-weight: bold;
        }

        </style>
        </head>
        <body>

        <!-- フレーム -->
        <div class="frame">
                <p>"""

        str3 = """</p>
        </div>
        </body>
        </html>"""

        ret_html = str1 + str_title + str2 + message + str3
        return ret_html


# HTML文字列を生成する関数（吹き出し）
def make_html_balloon(file_name, message, back_color='aliceblue'):

        str1 = """
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>

                /* Flexbox */
                .flex {
                display: flex;
                }

                .items-center {
                align-items: center;
                }

                /* 吹き出し（左） */
                .balloon-left {
                position: relative;
                display: inline-block;
                margin: 1.5em 0 1.5em 15px;
                padding: 1em;
                min-width: 120px;
                max-width: 100%;
                color: #555;
                font-size: 16px;
                background: """

        str2 = """;
                border-radius: 15px;
                border: solid 1px #888;
                }

                .balloon-left:before {
                content: "";
                position: absolute;
                top: 50%;
                left: 0;
                width: 10px;
                height: 10px;
                transform: translate(-50%, -50%) rotate(45deg);
                background: """
        
        str3 = """;
                border-left: solid 1px #888;
                border-bottom: solid 1px #888;
                }

                .balloon-left p {
                margin: 0;
                padding: 0;
                }

                </style>
                </head>
                <body>

                <!-- キャラクターと吹き出し -->
                <div class="flex items-center">
                <div>
                <img src="https://nai-lab.com/datasets/esports/balloon/"""

        str4 = """" alt="" style="width: 120px; height:auto;">
                </div>
                <div class="balloon-left">
                <p>"""

        str5 = """</p>
                </div>
                </div>

                </body>
                </html>"""

        ret_html = str1 + back_color + str2  + back_color + str3 + file_name + str4 + message + str5

        return ret_html