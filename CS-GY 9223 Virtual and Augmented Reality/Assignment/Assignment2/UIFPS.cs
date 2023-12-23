using UnityEngine;

public class UIFPS : MonoBehaviour
{
    public TextMesh tm_Fps;
    public TextMesh tm_AvgFps;


    private float _fCount;
    private float _fpsAccumulator = 0f;
    private float _fpsNextPeriod = 0f;


    private void Start()
    {
        _fpsNextPeriod = Time.realtimeSinceStartup + 1;
        _fCount = 0;
    }

    private void Update()
    {
        DisplayFPS();
        DisplayAvgFPS();
    }

    private void DisplayFPS()
    {
        float fps = 1f / Time.unscaledDeltaTime;
        tm_Fps.text = $"FPS (per frame): {fps}";
    }


    private void DisplayAvgFPS()
    {
        float currentFPS = 1f / Time.unscaledDeltaTime;
        _fpsAccumulator += currentFPS;
        _fCount += 1;
        if (Time.realtimeSinceStartup > _fpsNextPeriod)
        {
            float averageFPS = _fpsAccumulator / _fCount;
            tm_AvgFps.text = $"Average FPS (1 sec): {Mathf.RoundToInt(averageFPS)}";
            _fpsAccumulator = 0;
            _fpsNextPeriod += 1;
            _fCount = 0;
        }
    }
}