// ConMogu.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>
#include <stdlib.h>
#include <time.h>

enum CELL_STAT {
    EMPTY,      // モグラがいない
    EXIST,      // モグラがいる
    HIT,        // すでに叩かれたモグラがいる
    OUT_OF_CELL // 無効なセル
};

#define CELL_MAX    15
#define HIT_POINT   5
#define FAULT_POINT 3

int Random(int n)
{
    static int first = 0;
    if (first == 0) {
        srand((unsigned)time(NULL));
        first = 1; 
    }
    return (int)(rand() / (float)RAND_MAX * n);
}

class MoguraGame
{
public:
    MoguraGame(int moguraNo);
    void Show();
    CELL_STAT Hit(char c);
    int GetRestNo();
    int GetGain();
private:
    void ShowCell(int cellNo);

    CELL_STAT FieldCells[CELL_MAX];
    int MoguraNo;
    int RestNo;
    int Gain;
};

MoguraGame::MoguraGame(int moguraNo)
{
    MoguraNo = moguraNo;
    RestNo = moguraNo;
    Gain = 0;

    for (int n = 0; n < CELL_MAX; n++) {
        FieldCells[n] = EMPTY;
    }

    int cnt = 0;
    while (cnt < MoguraNo) {
        int ndx = Random(CELL_MAX);
        if (FieldCells[ndx] != EMPTY) continue;
        FieldCells[ndx] = EXIST;
        cnt++;
    }
}

int MoguraGame::GetRestNo()
{
    return RestNo;
}
int MoguraGame::GetGain()
{
    return Gain;
}

void MoguraGame::Show()
{
    for (int i = 0; i < CELL_MAX; i++) {
        if (i % 5 == 0) std::cout << "\n\n";
        ShowCell(i);

    }
    std::cout << "\t 残り:" << RestNo << "匹 得点:" << Gain << "点 ";
}

void MoguraGame::ShowCell(int cellNo)
{
    char c = 'a' + cellNo;
    if (FieldCells[cellNo] == HIT) c = '@';
    std::cout << ' ' << c << ' ';

}

CELL_STAT MoguraGame::Hit(char c)
{
    int no = tolower(c) - 'a';
    if (no < 0 || CELL_MAX <= no) return OUT_OF_CELL;
    CELL_STAT cellStat = FieldCells[no];
    if (cellStat == EXIST)
    {
        FieldCells[no] = HIT;
        RestNo--;
        Gain += HIT_POINT;

    }
    else if (cellStat == EMPTY)
        Gain -= FAULT_POINT;
    return cellStat;
}


int main()
{
    MoguraGame mogu(5);
    while (mogu.GetRestNo() > 0) {
        mogu.Show();
        char key;
        std::cout << "キーを売ってください：";
        std::cin >> key;
        if (mogu.Hit(key) == EXIST)
            std::cout << "☆ 成功 ☆" << std::endl;
        else
            std::cout << "★ 失敗 ★" << std::endl;
    }
    std::cout << "\nゲーム終了\n成績: " << mogu.GetGain() << "点" << std::endl;
    return 0;
    std::cout << "Hello World!\n";
}

// プログラムの実行: Ctrl + F5 または [デバッグ] > [デバッグなしで開始] メニュー
// プログラムのデバッグ: F5 または [デバッグ] > [デバッグの開始] メニュー

// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します
