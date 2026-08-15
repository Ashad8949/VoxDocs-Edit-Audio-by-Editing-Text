"""Migration: Add Translation, DubRender, and VoiceProfile models."""

from django.db import migrations, models
import django.db.models.deletion
import projects.models


class Migration(migrations.Migration):

    dependencies = [
        ('projects', '0001_initial'),
    ]

    operations = [
        # VoiceProfile model
        migrations.CreateModel(
            name='VoiceProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedding', models.JSONField(default=dict)),
                ('pitch_info', models.JSONField(default=dict)),
                ('spectral_features', models.JSONField(default=dict)),
                ('supported_languages', models.JSONField(default=list)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('project', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='voice_profile', to='projects.project')),
            ],
            options={
                'verbose_name_plural': 'Voice profiles',
            },
        ),
        # Translation model
        migrations.CreateModel(
            name='Translation',
            fields=[
                ('id', models.CharField(default=projects.models.new_id, editable=False, max_length=64, primary_key=True, serialize=False)),
                ('source_language', models.CharField(default='en', max_length=16)),
                ('target_language', models.CharField(max_length=16)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('translating', 'Translating'), ('ready', 'Ready'), ('failed', 'Failed')], default='pending', max_length=16)),
                ('error', models.TextField(blank=True, default='')),
                ('translated_text', models.JSONField(default=dict)),
                ('edits', models.JSONField(default=list)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='translations', to='projects.project')),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        # DubRender model
        migrations.CreateModel(
            name='DubRender',
            fields=[
                ('id', models.CharField(default=projects.models.new_id, editable=False, max_length=64, primary_key=True, serialize=False)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('rendering', 'Rendering'), ('ready', 'Ready'), ('failed', 'Failed')], default='pending', max_length=16)),
                ('error', models.TextField(blank=True, default='')),
                ('format', models.CharField(default='mp4', max_length=8)),
                ('file', models.CharField(blank=True, default='', max_length=255)),
                ('bytes', models.BigIntegerField(default=0)),
                ('duration', models.FloatField(default=0.0)),
                ('stats', models.JSONField(default=dict)),
                ('warnings', models.JSONField(default=list)),
                ('synthesis', models.JSONField(default=dict)),
                ('task_id', models.CharField(blank=True, default='', max_length=64)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('translation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='dub_renders', to='projects.translation')),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        # Add unique constraint on Translation
        migrations.AddConstraint(
            model_name='translation',
            constraint=models.UniqueConstraint(fields=['project', 'target_language'], name='unique_project_target_language'),
        ),
        # Add index on DubRender
        migrations.AddIndex(
            model_name='dubrender',
            index=models.Index(fields=['translation', '-created_at'], name='projects_du_translat_idx'),
        ),
    ]
