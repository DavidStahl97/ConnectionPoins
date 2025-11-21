using System;
using System.IO;
using System.Windows;
using Serilog;
using Serilog.Events;

namespace StepViewer;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        // Configure Serilog
        var logDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Logs");
        Directory.CreateDirectory(logDirectory);

        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .MinimumLevel.Override("Microsoft", LogEventLevel.Information)
            .Enrich.FromLogContext()
            .Enrich.WithProperty("Application", "StepViewer")
            .WriteTo.File(
                path: Path.Combine(logDirectory, "stepviewer-.log"),
                rollingInterval: RollingInterval.Day,
                outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}",
                retainedFileCountLimit: 30,
                fileSizeLimitBytes: 10_000_000)
            .WriteTo.Debug(
                outputTemplate: "{Timestamp:HH:mm:ss} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
            .CreateLogger();

        Log.Information("========================================");
        Log.Information("Application starting up");
        Log.Information("Version: {Version}", System.Reflection.Assembly.GetExecutingAssembly().GetName().Version);
        Log.Information("Log directory: {LogDirectory}", logDirectory);
        Log.Information("========================================");

        // Handle unhandled exceptions
        AppDomain.CurrentDomain.UnhandledException += OnUnhandledException;
        Current.DispatcherUnhandledException += OnDispatcherUnhandledException;
    }

    protected override void OnExit(ExitEventArgs e)
    {
        Log.Information("Application shutting down with exit code: {ExitCode}", e.ApplicationExitCode);
        Log.CloseAndFlush();
        base.OnExit(e);
    }

    private void OnUnhandledException(object sender, UnhandledExceptionEventArgs e)
    {
        var exception = e.ExceptionObject as Exception;
        Log.Fatal(exception, "Unhandled exception occurred. IsTerminating: {IsTerminating}", e.IsTerminating);

        MessageBox.Show(
            $"Ein schwerwiegender Fehler ist aufgetreten:\n\n{exception?.Message}\n\nDie Anwendung wird beendet.",
            "Schwerwiegender Fehler",
            MessageBoxButton.OK,
            MessageBoxImage.Error);
    }

    private void OnDispatcherUnhandledException(object sender, System.Windows.Threading.DispatcherUnhandledExceptionEventArgs e)
    {
        Log.Error(e.Exception, "Unhandled dispatcher exception occurred");

        MessageBox.Show(
            $"Ein Fehler ist aufgetreten:\n\n{e.Exception.Message}\n\nDetails wurden geloggt.",
            "Fehler",
            MessageBoxButton.OK,
            MessageBoxImage.Error);

        e.Handled = true; // Prevent application crash
    }
}

