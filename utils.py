import time

def initialize():
    print("Hello...")
    time.sleep(2)
    print("Everything is interlinked...")
    time.sleep(3)
    print("Dive into reality...")
    print("""
................................................................................................
......................................:=*#*=-:....::::::::.:=*##+-..............................
..................................=@+....::---**----------------::.-%#..........................
..............................-#+:.::------------++------------------::-**:.....................
...........................-%-..:----------------------------------------:.+#...................
.........................#=..:----------------------------------------------::#-................
.......................#-..:--------------------------------------------------::%:..............
.....................=+..:------------------------------------------------------.-#.............
....................*-.:---------------------------------------------------+-----::@............
...................#-.:----------------=------------------------------------#------.#:..........
..................+-.:------*---------=--------------------------------------#---:--:==.........
.................:+.:-:----+---------=--------------------------=-------------#---:--:==........
.................%::--:---+=::------=--:-------------------------#------=----:-#------:+-.......
.................+.:------#-:---------:-:-----:--*----------:----#--:-:-+----:-=+------:#:......
..............:-****-----*---------=------=------*----------:-----+----:-*------=-------.#......
............*::-=+++=+---#---------#------#------*----------------@==----+#------#------:+-.....
............*-++++++++=+*+----------------%------%#+---=----------#%------@%=----*-----#-.*.....
...........==+++++*+++=+#---------*-------%-+*+--@-----=----------#=*-----#+=-----@----*-:%.....
...........%==++++++++==#---------@-----+@@-----*#-----=----------*-+=----+:#-----@=---*=:#.....
...........:**=++++++=+=#---------#-*+--*-%-----*-*-----=---------#-=#----+:-*----@@---*%-*:....
.....:*%+=#%#*===++++++=*---------*-----*.#-----*.#-----=------=--%:.:*---+..#----@@---%@-#:....
...+*-#*:.=.::-==++=--==*---------%----#:.++++*#+.=-----=------+--#%%%@###@+.-=---##=--#+=%.....
.:%=#.....=:====+====-+=*---------@----*...*=+=#...#-----------=-=%@*++#=%%%%=%--===*-*:+=#.....
:%%.......:-=========-=-*---------@----:%%%@%%#%%%*=-----*----%--%@%%@##+*=.+%@--@==*-#.+*-.....
%=.........=-========-=-*---------@%--@%@@#**==#+.-*#----%----+----:@@@%#%+..%@-*==-+@..#%......
............++------=+--#---------%-##@@%%%%%+=%#....#-------#--*.:#@@%-=#=..#*++==-@...%.......
...............:*------*%---------%=@#*+#%@@@@=%%:....*---%--=-%..*-==----....+%==---%..........
................#:-----##---------@%%*:.#%@@@@#%%:.....-=--=+-#...-=:..:=+..:.=+==---%..........
................#:-----##---------@#%=:%=-=**=--*........#-+-=.....:+*%+......%==-----+.........
................+------+%---------%:+..==--...-+:.........@*%.................%===----%:........
...............---------*=--------+.:::..-#+*#-................=.............:%===-----#........
...............+--------+*---------.::::::::::..........................:....:@===-----*:.......
...............%:--------#---------*::::::::::...............................@+#===-----@.......
...............%:--------%---------@::::::::::.............:+#%:...........=%==#===------+......
..............:=---------*=---------@-::::.::.........:-------#:.........=%====+%===-----#:.....
..............%----------#*---------#=*#...............-==::..........-%+=======%===---%--#.....
..............*----------*%---------%====+%#=:....................-*%+==========+#==---#%-*:....
.............@----------=+**---------*=====#===#@@%=:........-%@*================@==---%#*#%*+..
............*----=------*==%---------#+====%=%=:..-*:--=%=@%%*+=+================#==-*#@#%%#%%@:
...........:+---=-------#==*=-=-------#====%#::.-:::-%++#=#+@-%+#=================*=-+@%#=%%%%%%
...........%---==------*====%--@------=+===*::.:-=::::::=:@+@-:*#=======+========+#**++#%%#*#%%@
..........@---==+------*====+#-=@------%==%-:.:::*-:::::--%%=%:%========++====+*#==:-+=++%@%%#%.
.........#---==*------+======**-@+*-----%*=:.::::#::::::--%%-+--+========#+#*+-=++-**#++#+@*:...
........==--===*------+=======*+*==#+----@:.::::*::::::::=*@-:=:%==+##%##=--*+-*#+==-@+=-*-.....
.......:*--===+=-----==========+%@====@---+.:::*--%+**+%-%+@-#-*@#@%=*=:=*=*%+======-%@==-%.....
.......%---===%------===========+%=====%%---::=*##%%##%%##@=-@**++**+*#+#%==========-#+*=-#:....
......:*--===#%-----=============+%*=+++-.:::::#%%#+@@%%%%##*=-+-+#=.:::**==========-#:%==-+....
......+---==++*-----==========+#+*%%%%%%%-.::::-%+*%@%%+===--+-+##*++====*=====++===-#.=+=-*....
......*---==@:=-----=======+=*+#%%%++++++%:.::=#*@#====--=-+#*-#+========#=====+*===-@..%=-#....
......*--==*:-=----==#=====**+%%%%#**+++++%:.*:=*:+#--+=%%-::-:%=========#=====##==--*..%+-*....
......=--==%..*----=+@=====%#%%%%%%%%%%%%+*%##=....=%%@%#==-::::+=======*+====+@*==-*...#+-+....
......:*-=*...%----=#@====+@#%%%%%%%%%@=*#=#*=#:.:+#%%*%#@-:+@#%%+======@+====%-+==-@...++#:....
.......%-=@...*=---=%+*===+##%%%%%%%%%%%=**=#@@@%%%%*#%%%%%=*%**#@=====#@====#:#==-#....**+.....
.......:#-@....#---=%.%===+**+%%%%%%%%%%@#%%%#%%%%#+#%%@@%%**%**@*#===*=#===#::%==#.....%#......
........==@....:+---%.:#==+*-%#%%%%%%%%%%%%%%%#%#@@**@****%***#**###=*=+*==%:.#==%......-.......
.........*@.....-+--#:.:%==%.@%%%%%%%%%%%+#%%%%%%##**%********#**##*@-.#=**..-*#+...............
..........:.......%-=+...*+*+%#%#%%%%+%%%%%%*#%%*#***#*****@***%***#**%*#...:@=.................
...................**#....:%%.%*%+%%%%%%+%%%%%%##****#*****%#***%**%+#=.........................
.....................+@.....:+.=%@@*#%%%%%%%+%#%#*****#*****#**%@#..............................
...................................+%##@###****%#*****@###%%+---%...............................
.......................................-%:=%@%%*-@+***+:*:::...=#*..............................
........................................%:.......*.......@*##%###%:.............................
........................................@#-:...:#.........%########.............................
.......................................*#######%-..........+%####%@.............................
.......................................%######%=............%#####*.............................
.......................................@######+...............+*:...............................
.......................................%%####%=.................................................
................................................................................................
................................................................................................
""")
    time.sleep(5)
    print("I'll merge you to the interlinked world...\n")
    time.sleep(3)