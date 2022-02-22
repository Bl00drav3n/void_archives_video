#define _CRT_SECURE_NO_WARNINGS
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <list>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <tesseract/baseapi.h>

#define WITH_VIDEO 0
#define WAIT_DELAY_MS 15

#define ARRAY_COUNT(x) (sizeof(x) / sizeof(x[0]))

enum event_type_t {
    EVENT_STIGMATA_SCREEN,
    EVENT_WEAPON_SCREEN,
    EVENT_DIVINE_KEY_SCREEN,
    EVENT_LINEUP_SCREEN,
    EVENT_ABYSS_BATTLE,
    EVENT_ARENA_BATTLE,

    EVENT_VALKYRIE_NAME,
    EVENT_VALKYRIE_RANK,
    EVENT_WEAPON,
    EVENT_STIGMATA,
    EVENT_ELF,
    EVENT_DIVINE_KEY,
};

struct event_t {
    event_type_t Type;
    std::string  Value;
};

struct state_t {
    int Width;
    int Height;
    cv::VideoCapture Capture;
    tesseract::TessBaseAPI Tess;
    std::list<event_t> Events;
    bool HadStigmataScreenIndicator;
    bool HadLineupScreenIndicator;
};

struct test_pixel_t {
    int x;
    int y;
    uchar Color[3];
};

struct rect_t {
    int X;
    int Y;
    int Width;
    int Height;
};

static void LOGMSG(const char* fmt, ...) {
    static FILE* LogFile = fopen("Log.txt", "w");
    if (LogFile) {
        va_list args;
        va_start(args, fmt);
        vfprintf(LogFile, fmt, args);
        va_end(args);
    }
}

static void OUTPUT(const char* fmt, ...) {
    static FILE* Out = stdout;
    if (Out) {
        va_list args;
        va_start(args, fmt);
        vfprintf(Out, fmt, args);
        fputc('\n', Out);
        va_end(args);
    }
}

static void log_timestamp(const cv::VideoCapture& Capture, const char* Msg) {
    int FrameNum = (int)Capture.get(cv::CAP_PROP_POS_FRAMES);
    int timer = (int)Capture.get(cv::CAP_PROP_POS_MSEC);
    int milliseconds = timer % 1000; timer /= 1000;
    int seconds = timer % 60; timer /= 60;
    int minutes = timer % 60; timer /= 60;
    int hours = timer;
    LOGMSG("Frame number %d (%d:%02d:%02d:%03d): %s\n", FrameNum, hours, minutes, seconds, milliseconds, Msg);
}

void add_event(state_t* State, event_type_t Type, const std::string& Value = std::string()) {
    event_t Event;
    Event.Type = Type;
    Event.Value = Value;
    State->Events.push_back(Event);
}

static float square(float x) {
    return x * x;
}

static int clamp(int a, int lower, int upper) {
    return std::min(std::max(a, lower), upper);
}

static std::string replace_char(char* Str, char ToReplace, char ReplaceWith) {
    char* Start = Str;
    while (*Str) {
        if (*Str == ToReplace) {
            *Str = ReplaceWith;
        }
        Str++;
    }
    return std::string(Start);
}

static std::string trim(const std::string &String) {
    const char* Str = String.c_str();
    while (Str[0] && isspace(Str[0])) {
        Str++;
    }
    const char* Ptr = Str;
    while (*Ptr++);
    while (Ptr > Str && !isspace(Ptr[0])) {
        Ptr--;
    }
    return std::string(Str, Ptr - Str);
}

static void draw_indicator(uchar *Pixels, int Width, int Height, int x, int y) {
    const uchar R = 0x00;
    const uchar G = 0xFF;
    const uchar B = 0x00;
    const int size = 32;
    int min_x = clamp(x - size / 2, 0, Width - 1);
    int max_x = clamp(x + size / 2, 0, Width - 1);
    int min_y = clamp(y - size / 2, 0, Height - 1);
    int max_y = clamp(y + size / 2, 0, Height - 1);
    for (int i = min_x; i < max_x; i++) {
        Pixels[3 * (Width * y + i) + 0] = B;
        Pixels[3 * (Width * y + i) + 1] = G;
        Pixels[3 * (Width * y + i) + 2] = R;
    }
    for (int i = min_y; i < max_y; i++) {
        Pixels[3 * (Width * i + x) + 0] = B;
        Pixels[3 * (Width * i + x) + 1] = G;
        Pixels[3 * (Width * i + x) + 2] = R;
    }
}

// TODO(rav3n): Provide a function pointer?
static void invert_image(uchar *Image, int Width, int Height, int Channels, int Stride) {
    for (int y = 0; y < Height; y++) {
        uchar* Row = Image + y * Stride;
        for (int x = 0; x < Width; x++) {
            for (int Channel = 0; Channel < Channels; Channel++) {
                uchar* Pixel = Row + Channels * x + Channel;
                *Pixel = 255 - *Pixel;
            }
        }
    }
}

static void change_contrast(uchar* Image, int Width, int Height, int Channels, int Stride, float Contrast) {
    for (int y = 0; y < Height; y++) {
        uchar* Row = Image + y * Stride;
        for (int x = 0; x < Width; x++) {
            for (int Channel = 0; Channel < Channels; Channel++) {
                uchar* Pixel = Row + Channels * x + Channel;
                float Value = Contrast * (Pixel[0] / 255.f - 1.f) + 1.f;
                *Pixel = clamp((int)((Value + 0.5f) * 255.f), 0x00, 0xff);
            }
        }
    }
}

static void to_grayscale(uchar* Image, int Width, int Height, int Channels, int Stride) {
    for (int y = 0; y < Height; y++) {
        uchar* Row = Image + y * Stride;
        for (int x = 0; x < Width; x++) {
            uchar* Pixel = Row + Channels * x;
            uchar Value = clamp((int)(0.299f * Pixel[2] + 0.587f * Pixel[1] + 0.114f * Pixel[0]), 0, 255);
            Pixel[0] = Pixel[1] = Pixel[2] = Value;
        }
    }
}

static bool screen_test(cv::Mat* Frame, cv::Size TargetSize, const test_pixel_t *TestPixels, int TestPixelCount, float ThresholdConfidence) {
    uchar* Pixels = Frame->ptr(0, 0);
    float Indicator = 0;
    for (int i = 0; i < TestPixelCount; i++) {
        const test_pixel_t* TestPixel = TestPixels + i;
        uchar* Pixel = Pixels + 3 * (TargetSize.width * TestPixel->y + TestPixel->x);
        uchar B = Pixel[0];
        uchar G = Pixel[1];
        uchar R = Pixel[2];

        float RDelta = 2.f * (0xff + R - TestPixel->Color[0]) / 510.f - 1.f;
        float GDelta = 2.f * (0xff + G - TestPixel->Color[1]) / 510.f - 1.f;
        float BDelta = 2.f * (0xff + B - TestPixel->Color[2]) / 510.f - 1.f;
        Indicator += sqrtf(square(RDelta) + square(GDelta) + square(BDelta)) / (3.f * TestPixelCount);
    }

    bool Result = (1.f - Indicator) >= ThresholdConfidence;
#if WITH_VIDEO
    if (Result) {
        for (int i = 0; i < TestPixelCount; i++) {
            const test_pixel_t* TestPixel = TestPixels + i;
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    draw_indicator(Pixels, TargetSize.width, TargetSize.height, TestPixel->x + x, TestPixel->y + y);
                }
            }
        }
    }
#endif

    return Result;
}

static void scan_stigmata_screen(state_t* State, cv::Mat* RefFrame) {
    add_event(State, EVENT_STIGMATA_SCREEN);

    static int StigmataFrameIndex = 0;
    log_timestamp(State->Capture, "Stigmata screen");

    uchar* Image = RefFrame->ptr(0, 0);
    const int ImageChannels = 3;
    const int ImageStrideBytes = ImageChannels * State->Width;

    const int NameBoxWidth = 484;
    const int NameBoxHeight = 72;
    rect_t NameBox = {
        188, 912, NameBoxWidth, NameBoxHeight
    };
    {
        rect_t* Box = &NameBox;
        uchar* SubImage = Image + ImageChannels * (State->Width * Box->Y + Box->X);
        change_contrast(SubImage, NameBoxWidth, NameBoxHeight, ImageChannels, ImageStrideBytes, 4.f);
        invert_image(SubImage, NameBoxWidth, NameBoxHeight, ImageChannels, ImageStrideBytes);
        to_grayscale(SubImage, NameBoxWidth, NameBoxHeight, ImageChannels, ImageStrideBytes);
        State->Tess.SetImage(SubImage, NameBoxWidth, NameBoxHeight, ImageChannels, ImageStrideBytes);
        State->Tess.Recognize(0);
        char* Str = State->Tess.GetUTF8Text();
        std::string Text = trim(replace_char(Str, '\n', ' '));
        delete[] Str;
        LOGMSG("Valkyrie: %s\n", Text.c_str());
        add_event(State, EVENT_VALKYRIE_NAME, Text);
        State->Tess.Clear();
    }

    const int StigmataBoxWidth = 284;
    const int StigmataBoxHeight = 188;
    char BoxNames[3] = { 'T', 'M', 'B' };
    rect_t StigmataBoxes[3] = {
        { 872, 550, StigmataBoxWidth, StigmataBoxHeight},
        {1232, 550, StigmataBoxWidth, StigmataBoxHeight},
        {1592, 550, StigmataBoxWidth, StigmataBoxHeight},
    };
    for (int i = 0; i < 3; i++) {
        rect_t* Box = StigmataBoxes + i;
        uchar* SubImage = Image + ImageChannels * (State->Width * Box->Y + Box->X);
        invert_image(SubImage, StigmataBoxWidth, StigmataBoxHeight, ImageChannels, ImageStrideBytes);
        change_contrast(SubImage, StigmataBoxWidth, StigmataBoxHeight, ImageChannels, ImageStrideBytes, 4.f);
        State->Tess.SetImage(SubImage, StigmataBoxWidth, StigmataBoxHeight, ImageChannels, ImageStrideBytes);
        State->Tess.Recognize(0);
        char* Str = State->Tess.GetUTF8Text();
        std::string Text = trim(replace_char(Str, '\n', ' '));
        delete[] Str;
        LOGMSG("Stigmata (%c): %s\n", BoxNames[i], Text.c_str());
        add_event(State, EVENT_STIGMATA, Text);
        State->Tess.Clear();
    }

    char Buffer[256];
    snprintf(Buffer, sizeof(Buffer), "%s/stigmata_frame_%d.png", "./Output", StigmataFrameIndex++);
    cv::imwrite(Buffer, *RefFrame);
}

static void scan_lineup_screen(state_t* State, cv::Mat* RefFrame) {
    add_event(State, EVENT_LINEUP_SCREEN);

    static int LineupFrameIndex = 0;
    log_timestamp(State->Capture, "Lineup screen");

    uchar* Image = RefFrame->ptr(0, 0);
    const int ImageChannels = 3;
    const int ImageStrideBytes = ImageChannels * State->Width;

    char Buffer[256];
    snprintf(Buffer, sizeof(Buffer), "%s/lineup_frame_%d.png", "./Output", LineupFrameIndex++);
    cv::imwrite(Buffer, *RefFrame);
}

static void output_events(const state_t *State) {
    for (auto it = State->Events.begin(); it != State->Events.end(); it++) {
        const event_t& Event = *it;
        const char* Value = Event.Value.c_str();
        switch (Event.Type) {
        case EVENT_STIGMATA_SCREEN:
            OUTPUT("[STIGMATA_SCREEN]");
            break;
        case EVENT_STIGMATA:
            OUTPUT("Stigmata=%s", Value);
            break;
        case EVENT_VALKYRIE_NAME:
            OUTPUT("Valkyrie=%s", Value);
            break;
        case EVENT_LINEUP_SCREEN:
            OUTPUT("[LINEUP_SCREEN]");
            break;
        default:
            LOGMSG("Type %d not implemented!");
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        LOGMSG("Expected 1 argument but got %d\n", argc - 1);
        return 0;
    }

    state_t State;
    State.Width = 1920;
    State.Height = 1080;
    State.HadStigmataScreenIndicator = false;

    if (State.Tess.Init(".", "eng")) {
        LOGMSG("Could not initialize tesseract\n");
        return -1;
    }

    State.Tess.SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK);
    State.Tess.SetVariable("save_best_choices", "T");
    State.Tess.SetVariable("user_defined_dpi", "300");

    LOGMSG("Initialized tesseract %s %s\n", State.Tess.Version(), State.Tess.GetInitLanguagesAsString());

    std::string SrcFile = argv[1];
    State.Capture.open(SrcFile);
    if (!State.Capture.isOpened()) {
        LOGMSG("Could not open file %s\n", SrcFile.c_str());
        return -1;
    }

    LOGMSG("Streaming video file from %s\n", SrcFile.c_str());

#if WITH_VIDEO
    const char* WinName = "Test";
    cv::namedWindow(WinName, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(WinName, 0, 0);
#endif

    const float StigmataScreenThresholdConfidence = 0.97f;
    const test_pixel_t StigmataScreenIndicators[] = {
        {  120, 200, 0xee, 0x9a, 0xff },
        {  990, 864, 0xff, 0xdd, 0x47 },
        { 1350, 864, 0xff, 0xdd, 0x47 },
        { 1710, 864, 0xff, 0xdd, 0x47 },
        { 1280, 974, 0x00, 0xc9, 0xff },
    };

    const float LineupScreenThresholdConfidence = 0.97f;
    const test_pixel_t LineupScreenIndicators[] = {
        { 1762, 168, 0xff, 0xdd, 0x47 },
        { 1762, 390, 0xff, 0xdd, 0x47 },
        { 1762, 608, 0xff, 0xdd, 0x47 },
        {  181,  97, 0xff, 0xdb, 0x48 },
        { 1520, 986, 0x00, 0x5a, 0x7e },
    };

    const cv::Size TargetSize = cv::Size(State.Width, State.Height);
    LOGMSG("Framerate: %d\nFrame count: %d\n", (int)State.Capture.get(cv::CAP_PROP_FPS), (int)State.Capture.get(cv::CAP_PROP_FRAME_COUNT));
    LOGMSG("Stigmata screen threshold confidence value: %.6f\n", StigmataScreenThresholdConfidence);
    LOGMSG("Lineup screen threshold confidence value: %.6f\n", LineupScreenThresholdConfidence);

    cv::Mat Frame;
    for (State.Capture >> Frame; !Frame.empty(); State.Capture >> Frame) {
        cv::Mat ResizedFrame;
        cv::Mat* RefFrame = &Frame;
        assert(Frame.isContinuous());
        if (Frame.isContinuous()) {
            if (Frame.size() != TargetSize) {
                cv::resize(Frame, ResizedFrame, TargetSize);
                RefFrame = &ResizedFrame;
            }

            bool check = true;
            if (check && screen_test(RefFrame, TargetSize, StigmataScreenIndicators, ARRAY_COUNT(StigmataScreenIndicators), StigmataScreenThresholdConfidence)) {
                check = false;
                if (!State.HadStigmataScreenIndicator) {
                    State.HadStigmataScreenIndicator = true;
                    scan_stigmata_screen(&State, RefFrame);
                }
            }
            else {
                State.HadStigmataScreenIndicator = false;
            }

            if (check && screen_test(RefFrame, TargetSize, LineupScreenIndicators, ARRAY_COUNT(LineupScreenIndicators), LineupScreenThresholdConfidence)) {
                check = false;
                if (!State.HadLineupScreenIndicator) {
                    State.HadLineupScreenIndicator = true;
                    scan_lineup_screen(&State, RefFrame);
                }
            }
            else {
                State.HadLineupScreenIndicator = false;
            }
        }
#if WITH_VIDEO
        char c = (char)cv::waitKey(WAIT_DELAY_MS);
        if (c == 27) return 0;
        cv::imshow(WinName, *RefFrame);
#endif
    }

    output_events(&State);

    return 0;
}