#include "Util.h"
#include <string.h>

#include "windows.h"
#include "timeapi.h"

namespace Sun {

    //FIXME
    std::string getGlobalPath() {
        //FIXME Platform path

        //WCHAR exePath[256];
        //::GetModuleFileName(NULL, exePath, 255);

        //去掉执行的文件名。
        //(strrchr(exePath, '\\'))[1] = 0;
        //printf(exePath)

        return "D:\\GIT\\testproj\\Sun\\Render";

    }

    int sys_timeBase = 0;

    int Sys_Milliseconds()
    {
        int			sys_curtime;
        static bool	initialized = false;

        if (!initialized) {
            sys_timeBase = ::timeGetTime();
            initialized = true;
        }
        sys_curtime = ::timeGetTime() - sys_timeBase;

        return sys_curtime;
    }

}